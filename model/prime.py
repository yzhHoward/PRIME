import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import length_to_mask


class FeatureRegression(nn.Module):
    def __init__(self, input_size, out_size=None):
        super(FeatureRegression, self).__init__()
        if out_size is None:
            out_size = input_size
        self.W = nn.Parameter(torch.Tensor(out_size, input_size))
        self.b = nn.Parameter(torch.Tensor(out_size))
        m = torch.ones(out_size, input_size) - torch.eye(out_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class Rits(nn.Module):
    def __init__(self, x_dim, hidden_dim, demo_dim, reverse=False):
        super(Rits, self).__init__()
        self.hidden_dim = hidden_dim
        self.reverse = reverse
        self.rnn_cell = nn.GRUCell(x_dim * 2 + demo_dim, hidden_dim)

        self.temp_decay_h = nn.Linear(x_dim, hidden_dim)
        stdv = 1. / math.sqrt(hidden_dim)
        self.temp_decay_h.weight.data.uniform_(-stdv, stdv)

        self.hist_reg = nn.Linear(hidden_dim, x_dim)
        self.feat_reg = FeatureRegression(x_dim, x_dim)

        self.proto_attn = ProtoAttn(hidden_dim)
        self.W = nn.Linear(2*hidden_dim, 1)

    def forward(self, values, lens, static, times, masks, prototype, proto):
        h = torch.zeros((values.shape[0], self.hidden_dim), device='cuda')
        max_len = lens[0].item()
        x_h = torch.zeros((values.shape[0], values.shape[2]), device='cuda')
        hs = []
        x_cs = []
        z_cs = []
        lens = lens.unsqueeze(1)
        gamma_h = torch.exp(-F.relu(self.temp_decay_h(times)))
        for t in range(lens[0]):
            if self.reverse:
                t = max_len - t - 1
            x = values[:, t, :]
            m = masks[:, t, :]
            if proto:
                h1 = self.proto_attn(h, prototype)
                alpha = torch.sigmoid(self.W(torch.cat((h, h1), dim=1)))
                h = alpha * h + (1 - alpha) * h1
            h = h * gamma_h[:, t, :]
            x_h = torch.where((t < lens), self.hist_reg(h), x_h)
            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            z_c = m * x + (1 - m) * z_h
            inputs = torch.cat([z_c, m, static], dim=1)
            h_new = self.rnn_cell(inputs, h)
            h = torch.where((t < lens), h_new, h)
            hs.append(h)
            x_cs.append(x_c)
            z_cs.append(z_c)
        hs = torch.stack(hs, dim=1)
        x_cs = torch.stack(x_cs, dim=1)
        z_cs = torch.stack(z_cs, dim=1)
        return hs, h, [x_cs, z_cs]


class ProtoAttn(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, proto_feat):
        query = self.Q(h).unsqueeze(1)  # b, 1, h
        key = self.K(proto_feat)  # p, h
        value = self.V(proto_feat)  # p, h
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(proto_feat.shape[-2])  # b, 1, p
        attn = F.softmax(score, dim=2)  # b, t, p
        attn = F.dropout(attn, 0.3)
        out = torch.matmul(attn, value).squeeze(1)  # b, 1, h
        return out


class ProtoAttn1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Q1 = nn.Linear(hidden_dim, hidden_dim)
        self.K1 = nn.Linear(hidden_dim, hidden_dim)
        self.V1 = nn.Linear(hidden_dim, hidden_dim)

        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hs, proto_feat, lens, mask=None):
        query = self.Q1(proto_feat).unsqueeze(0)  # p, h
        key = self.K1(hs)  # b, t, h
        value = self.V1(hs)  # b, t, h
        attn = F.softmax(torch.matmul(query, key.transpose(1, 2) / torch.sqrt(lens)).masked_fill(mask.transpose(1, 2), -1e9), dim=2)  # b, p, t
        proto_feat = torch.matmul(attn, value)  # b, p, h

        query = self.Q(hs)  # b, t, h
        key = self.K(proto_feat)  # b, p, h
        value = self.V(proto_feat)  # b, p, h
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(proto_feat.shape[-2])
        attn = F.softmax(score, dim=2)  # b, t, p
        # attn = F.dropout(attn, 0.3)
        out = torch.matmul(attn, value)  # b, t, h
        return out


class Prime(nn.Module):
    def __init__(self, input_dim, demo_dim, hidden_dim=32, proto_num=50, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_f = Rits(input_dim, hidden_dim, demo_dim)
        self.gru_b = Rits(input_dim, hidden_dim, demo_dim, reverse=True)
        self.prototype = nn.Parameter(torch.empty((proto_num, 2*hidden_dim), dtype=torch.float32, device='cuda'))
        self.proto_attn = ProtoAttn1(2*hidden_dim)
        self.predict_obs = nn.Sequential(
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, x, lens, static, times, times_rev, mask, proto=True):
        # hs_f, ht_f, impute_f = self.gru_f(x, lens, static, times, mask, None)
        # hs_b, ht_b, impute_b = self.gru_b(x, lens, static, times_rev, mask, None)
        hs_f, ht_f, impute_f = self.gru_f(x, lens, static, times, mask, self.prototype[:, :self.hidden_dim], proto)
        hs_b, ht_b, impute_b = self.gru_b(x, lens, static, times_rev, mask, self.prototype[:, self.hidden_dim:], proto)
        hs_b = torch.fliplr(hs_b)
        for i in range(len(impute_b)):
            impute_b[i] = torch.fliplr(impute_b[i])
        hs = torch.cat((hs_f, hs_b), dim=2)  # b, t, 2h
        mask1 = length_to_mask(lens, lens[0], dtype=torch.float32, device='cuda') == 0 # b, t
        if proto:
            hs1 = self.proto_attn(hs, self.prototype, lens.reshape(-1, 1, 1), mask1.unsqueeze(2))  # b, 2d, h
        else:
            hs1 = torch.zeros_like(hs, device=hs.device)
        out = self.predict_obs(torch.cat((hs, hs1), dim=2))
        # out = self.predict_obs(hs)
        return hs, impute_f + impute_b, out, None
