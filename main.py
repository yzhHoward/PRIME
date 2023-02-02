import argparse
import json
import logging
import math
import os
import random
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy

from data.challenge import load_challenge_2019
from data.challenge2012 import load_challenge_2012
from data.person_activity import load_person_activity
from model.prime import Prime
from util.metrics import print_metrics_binary, bootstrap, mean_squared_error, mean_absolute_error, bootstrap_regression
from util.utils import batch_iter, pad_sents
from util.log import init_logging


def test(args, checkpoint_path, test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path))
    save_epoch = checkpoint['epoch']
    logging.info("last saved model is in epoch {}".format(save_epoch))
    enc.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    enc.eval()
    test_loss = 0
    test_recon_loss = 0
    x_pred = []
    x_true = []
    x_mask = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for step, batch_train in enumerate(batch_iter((test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing), batch_size=args.batch_size, shuffle=True), 1):
            batch_x, batch_y, batch_x_len, batch_static, batch_x_mask, batch_times, batch_times_rev, batch_x_missing, batch_mask_missing = batch_train
            batch_x = torch.tensor(pad_sents(batch_x, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_y = torch.tensor(batch_y, dtype=torch.float32, device='cuda')
            batch_x_len = torch.tensor(batch_x_len, dtype=torch.long, device='cuda')
            batch_static = torch.tensor(batch_static, dtype=torch.float32, device='cuda')
            batch_x_mask = torch.tensor(pad_sents(batch_x_mask, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times = torch.tensor(pad_sents(batch_times, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times_rev = torch.tensor(pad_sents(batch_times_rev, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_x_missing = torch.tensor(pad_sents(batch_x_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_mask_missing = torch.tensor(pad_sents(batch_mask_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            if args.static:
                batch_static = batch_static.unsqueeze(1).repeat(1, batch_x_len[0], 1)
                batch_x = torch.cat((batch_x, batch_static), dim=2)
                batch_x_missing = torch.cat((batch_x_missing, batch_static), dim=2)
                mask = torch.ones((batch_x_len.shape[0], batch_x_len[0], args.demo_dim)).cuda()
                batch_x_mask = torch.cat((batch_x_mask, mask - 1), dim=2)
                batch_mask_missing = torch.cat((batch_mask_missing, mask), dim=2)
                mask[:, 0, :] = 0
                batch_times = torch.cat((batch_times, mask), dim=2)
                mask[:, 0, :] = 1
                mask[torch.arange(mask.shape[0]), batch_x_len - 1] = 0
                batch_times_rev = torch.cat((batch_times_rev, mask), dim=2)
            hs, imputes, x_hat, y_hat = enc(batch_x_missing, batch_x_len, batch_static, batch_times, batch_times_rev, batch_mask_missing, save_epoch >= args.proto_epoch)
            recon_loss = mean_squared_error(batch_x, x_hat, batch_x_mask)
            loss = recon_loss
            for hx in imputes:
                loss += torch.sum(torch.abs(batch_x - hx) * batch_x_mask) / (torch.sum(batch_x_mask) + 1e-5) * 0.3
            test_loss += loss.item() * batch_y.shape[0]
            test_recon_loss += recon_loss.item() * batch_y.shape[0]
            x_pred += x_hat.reshape(-1, args.x_dim).detach()
            x_true += batch_x.reshape(-1, args.x_dim).detach()
            x_mask += batch_x_mask.reshape(-1, args.x_dim).detach()
    logging.info('Test Loss %.4f, Test Recon Loss %.4f' % (
        test_loss / len(test_y), test_recon_loss / len(test_y)))
    x_true, x_pred, x_mask = torch.stack(x_true), torch.stack(x_pred), torch.stack(x_mask)
    logging.info('MSE %.4f, MAE %.4f' % (
        mean_squared_error(x_true, x_pred, x_mask), mean_absolute_error(x_true, x_pred, x_mask)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='c12', choices=['c12', 'c19', 'activity'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--x-dim', type=int, default=34)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--demo-dim', type=int, default=5)
    parser.add_argument('--output-dim', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--proto_num', type=int, default=64)
    parser.add_argument('--missing', type=float, default=0.9)
    parser.add_argument('--margin', type=float, default=50)
    parser.add_argument('--static', type=bool, default=False)
    parser.add_argument('--proto_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', type=str,
                        default='./export/c12/impute3-e25-seed42')
    args = parser.parse_args()
    if args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_root = logging.getLogger()
    init_logging(log_root, args.save_dir if args.save_model else None)
    logging.info(json.dumps(vars(args), indent=4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'c12':
        args.x_dim = 37
        args.demo_dim = 4
        (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_x_missing, train_mask_missing), (
            val_x, val_y, val_x_len, val_static, val_x_mask, val_times, val_times_rev, val_x_missing, val_mask_missing), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing) = load_challenge_2012(time_gap=True, ratio=args.missing)
    elif args.dataset == 'c19':
        args.x_dim = 34
        args.demo_dim = 5
        (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_x_missing, train_mask_missing), (
            val_x, val_y, val_x_len, val_static, val_x_mask, val_times, val_times_rev, val_x_missing, val_mask_missing), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing) = load_challenge_2019(time_gap=True, ratio=args.missing)
    elif args.dataset == 'activity':
        args.x_dim = 12
        args.demo_dim = 0
        (train_x, train_y, train_x_len, train_x_mask, train_times, train_times_rev, train_x_missing, train_mask_missing), (
            val_x, val_y, val_x_len, val_x_mask, val_times, val_times_rev, val_x_missing, val_mask_missing), (
            test_x, test_y, test_x_len, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing) = load_person_activity(time_gap=True, ratio=args.missing)
        train_static, val_static, test_static = np.zeros((train_x.shape[0], 0)), np.zeros((val_x.shape[0], 0)), np.zeros((test_x.shape[0], 0))
    else:
        raise Exception("Dataset not exist!")
    logging.info('Dataset Loaded.')
    pad_token_x = np.zeros(args.x_dim)
    max_length = max(max(train_x_len), max(val_x_len), max(test_x_len))

    if args.static:
        args.x_dim += args.demo_dim
    enc = Prime(args.x_dim, args.demo_dim, args.hidden_dim, args.proto_num).cuda()
    params = enc.parameters()
    optimizer = torch.optim.Adam(params, args.lr)

    # proto init
    with torch.no_grad():
        proto = []
        for batch_train in batch_iter((train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_x_missing, train_mask_missing), batch_size=args.batch_size):
            batch_x, batch_y, batch_x_len, batch_static, batch_x_mask, batch_times, batch_times_rev, batch_x_missing, batch_mask_missing = batch_train
            batch_x = torch.tensor(pad_sents(batch_x, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_y = torch.tensor(batch_y, dtype=torch.float32, device='cuda')
            batch_x_len = torch.tensor(batch_x_len, dtype=torch.long, device='cuda')
            batch_static = torch.tensor(batch_static, dtype=torch.float32, device='cuda')
            batch_x_mask = torch.tensor(pad_sents(batch_x_mask, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times = torch.tensor(pad_sents(batch_times, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times_rev = torch.tensor(pad_sents(batch_times_rev, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            len_mask = (torch.arange(batch_x_len[0], device='cuda', dtype=torch.int).expand(
                        len(batch_x_len), batch_x_len[0]) < batch_x_len.unsqueeze(1)).to(torch.bool).unsqueeze(-1)
            batch_x_missing = torch.tensor(pad_sents(batch_x_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_mask_missing = torch.tensor(pad_sents(batch_mask_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            if args.static:
                batch_static = batch_static.unsqueeze(1).repeat(1, batch_x_len[0], 1)
                batch_x = torch.cat((batch_x, batch_static), dim=2)
                batch_x_missing = torch.cat((batch_x_missing, batch_static), dim=2)
                mask = torch.ones((batch_x_len.shape[0], batch_x_len[0], args.demo_dim)).cuda()
                batch_x_mask = torch.cat((batch_x_mask, mask - 1), dim=2)
                batch_mask_missing = torch.cat((batch_mask_missing, mask), dim=2)
                mask[:, 0, :] = 0
                batch_times = torch.cat((batch_times, mask), dim=2)
                mask[:, 0, :] = 1
                mask[torch.arange(mask.shape[0]), batch_x_len - 1] = 0
                batch_times_rev = torch.cat((batch_times_rev, mask), dim=2)
            proto.append(enc(batch_x_missing, batch_x_len, batch_static, batch_times, batch_times_rev, batch_mask_missing)[0].masked_select(len_mask).reshape(-1, 2*args.hidden_dim).cpu())
        proto = torch.cat((proto), dim=0)
        kmeans = MiniBatchKMeans(n_clusters=args.proto_num)
        kmeans.fit(proto.reshape(proto.shape[0], -1).numpy())
        enc.prototype.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device='cuda'))
        del proto, kmeans

    best_auc = 0
    best_prc = 0
    best_mse = 1
    for i in range(0, args.epochs):
        train_loss = 0
        val_loss = 0
        val_recon_loss = 0
        x_pred = []
        x_true = []
        x_mask = []
        y_pred = []
        y_true = []
        enc.train()
        for step, batch_train in enumerate(batch_iter((train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_x_missing, train_mask_missing), batch_size=args.batch_size, shuffle=True), 1):
            batch_x, batch_y, batch_x_len, batch_static, batch_x_mask, batch_times, batch_times_rev, batch_x_missing, batch_mask_missing = batch_train
            batch_x = torch.tensor(pad_sents(batch_x, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_y = torch.tensor(batch_y, dtype=torch.float32, device='cuda')
            batch_x_len = torch.tensor(batch_x_len, dtype=torch.long, device='cuda')
            batch_static = torch.tensor(batch_static, dtype=torch.float32, device='cuda')
            batch_x_mask = torch.tensor(pad_sents(batch_x_mask, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times = torch.tensor(pad_sents(batch_times, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_times_rev = torch.tensor(pad_sents(batch_times_rev, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            len_mask = (torch.arange(batch_x_len[0], device='cuda', dtype=torch.int).expand(
                        len(batch_x_len), batch_x_len[0]) < batch_x_len.unsqueeze(1)).to(torch.bool).unsqueeze(-1)
            batch_x_missing = torch.tensor(pad_sents(batch_x_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            batch_mask_missing = torch.tensor(pad_sents(batch_mask_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
            if args.static:
                batch_static = batch_static.unsqueeze(1).repeat(1, batch_x_len[0], 1)
                batch_x = torch.cat((batch_x, batch_static), dim=2)
                batch_x_missing = torch.cat((batch_x_missing, batch_static), dim=2)
                mask = torch.ones((batch_x_len.shape[0], batch_x_len[0], args.demo_dim)).cuda()
                batch_x_mask = torch.cat((batch_x_mask, mask - 1), dim=2)
                batch_mask_missing = torch.cat((batch_mask_missing, mask), dim=2)
                mask[:, 0, :] = 0
                batch_times = torch.cat((batch_times, mask), dim=2)
                mask[:, 0, :] = 1
                mask[torch.arange(mask.shape[0]), batch_x_len - 1] = 0
                batch_times_rev = torch.cat((batch_times_rev, mask), dim=2)
            hs, imputes, x_hat, y_hat = enc(batch_x_missing, batch_x_len, batch_static, batch_times, batch_times_rev, batch_mask_missing, i >= args.proto_epoch)
            recon_loss = mean_squared_error(batch_x, x_hat, batch_x_mask)
            loss = 0 + recon_loss
            for hx in imputes:
                loss += torch.sum(torch.abs(batch_x - hx) * batch_x_mask) / (torch.sum(batch_x_mask) + 1e-5) * 0.3

            # proto loss
            len_mask = (torch.arange(batch_x_len[0], device='cuda', dtype=torch.int).expand(
                len(batch_x_len), batch_x_len[0]) < batch_x_len.unsqueeze(1)).to(torch.bool).unsqueeze(-1)
            hs = hs.masked_select(len_mask).reshape(-1, 2*args.hidden_dim).detach()
            distance = torch.cdist(enc.prototype, hs)  # p, b
            rank = linear_sum_assignment(distance.detach().cpu().numpy())
            l2_loss = torch.clamp(args.margin / math.sqrt(args.proto_num) - torch.pdist(enc.prototype), 0).mean()
            proto_loss = 0.1 * distance[rank[0], rank[1]].mean() + 1 * torch.min(distance, dim=0)[0].mean() + 0.1 * l2_loss
            loss += proto_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.shape[0]
            if step % 40 == 0:
                logging.info('Epoch %d, Step %d: Loss %.4f, Recon Loss %.4f, Proto Loss %.4f' % (
                    i, step, loss.item(), recon_loss.item(), 0))

        enc.eval()
        with torch.no_grad():
            for step, batch_train in enumerate(batch_iter((val_x, val_y, val_x_len, val_static, val_x_mask, val_times, val_times_rev, val_x_missing, val_mask_missing), batch_size=args.batch_size, shuffle=True), 1):
                batch_x, batch_y, batch_x_len, batch_static, batch_x_mask, batch_times, batch_times_rev, batch_x_missing, batch_mask_missing = batch_train
                batch_x = torch.tensor(pad_sents(batch_x, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                batch_y = torch.tensor(batch_y, dtype=torch.float32, device='cuda')
                batch_x_len = torch.tensor(batch_x_len, dtype=torch.long, device='cuda')
                batch_static = torch.tensor(batch_static, dtype=torch.float32, device='cuda')
                batch_x_mask = torch.tensor(pad_sents(batch_x_mask, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                batch_times = torch.tensor(pad_sents(batch_times, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                batch_times_rev = torch.tensor(pad_sents(batch_times_rev, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                batch_x_missing = torch.tensor(pad_sents(batch_x_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                batch_mask_missing = torch.tensor(pad_sents(batch_mask_missing, pad_token_x, batch_x_len[0]), dtype=torch.float32, device='cuda')
                if args.static:
                    batch_static = batch_static.unsqueeze(1).repeat(1, batch_x_len[0], 1)
                    batch_x = torch.cat((batch_x, batch_static), dim=2)
                    batch_x_missing = torch.cat((batch_x_missing, batch_static), dim=2)
                    mask = torch.ones((batch_x_len.shape[0], batch_x_len[0], args.demo_dim)).cuda()
                    batch_x_mask = torch.cat((batch_x_mask, mask - 1), dim=2)
                    batch_mask_missing = torch.cat((batch_mask_missing, mask), dim=2)
                    mask[:, 0, :] = 0
                    batch_times = torch.cat((batch_times, mask), dim=2)
                    mask[:, 0, :] = 1
                    mask[torch.arange(mask.shape[0]), batch_x_len - 1] = 0
                    batch_times_rev = torch.cat((batch_times_rev, mask), dim=2)
                hs, imputes, x_hat, y_hat = enc(batch_x_missing, batch_x_len, batch_static, batch_times, batch_times_rev, batch_mask_missing, i >= args.proto_epoch)
                recon_loss = mean_squared_error(batch_x, x_hat, batch_x_mask)
                loss = recon_loss
                for hx in imputes:
                    loss += torch.sum(torch.abs(batch_x - hx) * batch_x_mask) / (torch.sum(batch_x_mask) + 1e-5) * 0.3
                val_loss += loss.item() * batch_y.shape[0]
                val_recon_loss += recon_loss.item() * batch_y.shape[0]
                x_pred += x_hat.reshape(-1, args.x_dim).detach()
                x_true += batch_x.reshape(-1, args.x_dim).detach()
                x_mask += batch_x_mask.reshape(-1, args.x_dim).detach()

        logging.info('Epoch %d: Train Loss %.4f, Valid Loss %.4f, Valid Recon Loss %.4f' %
                     (i, train_loss / len(train_y), val_loss / len(val_y), val_recon_loss / len(val_y)))
        x_true, x_pred, x_mask = torch.stack(x_true), torch.stack(x_pred), torch.stack(x_mask)
        cur_mse = mean_squared_error(x_true, x_pred, x_mask)
        if cur_mse < best_mse:
            best_mse = cur_mse
            state = {
                'model': enc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i
            }
            logging.info('----- Save best model - MSE: %.4f -----' %
                         cur_mse)
            torch.save(state, os.path.join(args.save_dir, 'checkpoint-mse.pth'))

    test(args, 'checkpoint-mse.pth', test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_x_missing, test_mask_missing)
