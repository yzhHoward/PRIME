import os
import numpy as np
from copy import deepcopy
import torch
import pickle


def load_electricity(time_gap = True, ratio = 1, all = False):
    x, mask = pickle.load(open('./data/Electricity/data.pkl', 'rb'))
    x_len = np.full((x.shape[0],), x.shape[1])
    t = np.ones((x.shape[0], x.shape[1]))
    y = np.zeros((x.shape[0],))

    missing_x = deepcopy(x)
    missing_mask = deepcopy(mask)
    random_mask = np.random.rand(*missing_mask.shape) > ratio
    zero = np.zeros_like(missing_x)
    missing_x = np.where(~random_mask, missing_x, zero)
    missing_mask = np.where(~random_mask, missing_mask, zero)
    mask = np.where(random_mask, mask, zero)
    
    if time_gap:
        t = np.expand_dims(t, -1)
        time = []
        time_rev = []
        gap = np.zeros((missing_mask.shape[0], missing_mask.shape[2]))
        gap_rev = np.zeros((missing_mask.shape[0], missing_mask.shape[2]))
        for i in range(missing_mask.shape[1]):
            time.append(gap)
            time_rev.append(gap_rev)
            gap = np.where(missing_mask[:, i] > 0, t[:, i], gap + t[:, i])
            rev = missing_mask.shape[1] - i - 1
            gap_rev = np.where(missing_mask[:, rev] > 0, t[:, rev], gap_rev + t[:, rev])
        time = np.stack(time, 1)
        time_rev = np.stack(time_rev, 1)
    else:
        time = t / np.max(t)
    if all:
        t = np.squeeze(t, -1)
        time_nogap = t / np.max(t)

    index = torch.randperm(len(x)).numpy()
    train_num = int(len(x) * 0.8)
    dev_num = int(len(x) * 0.1)
    test_num = len(x) - train_num - dev_num

    train_x = x[index[:train_num]]
    train_y = y[index[:train_num]]
    train_x_len = x_len[index[:train_num]]
    train_x_mask = mask[index[:train_num]]
    train_times = time[index[:train_num]]
    if time_gap:
        train_times_rev = time_rev[index[:train_num]]
    train_missing_x = missing_x[index[:train_num]]
    train_missing_mask = missing_mask[index[:train_num]]
    if all:
        train_times_nogap = time_nogap[index[:train_num]]

    dev_x = x[index[train_num:train_num+dev_num]]
    dev_y = y[index[train_num:train_num+dev_num]]
    dev_x_len = x_len[index[train_num:train_num+dev_num]]
    dev_x_mask = mask[index[train_num:train_num+dev_num]]
    dev_times = time[index[train_num:train_num+dev_num]]
    if time_gap:
        dev_times_rev = time_rev[index[train_num:train_num+dev_num]]
    dev_missing_x = missing_x[index[train_num:train_num+dev_num]]
    dev_missing_mask = missing_mask[index[train_num:train_num+dev_num]]
    if all:
        dev_times_nogap = time_nogap[index[train_num:train_num+dev_num]]

    test_x = x[index[-test_num:]]
    test_y = y[index[-test_num:]]
    test_x_len = x_len[index[-test_num:]]
    test_x_mask = mask[index[-test_num:]]
    test_times = time[index[-test_num:]]
    if time_gap:
        test_times_rev = time_rev[index[-test_num:]]
    test_missing_x = missing_x[index[-test_num:]]
    test_missing_mask = missing_mask[index[-test_num:]]
    if all:
        test_times_nogap = time_nogap[index[-test_num:]]

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    if time_gap:
        if all:
            return (train_x, train_y, train_x_len, train_x_mask, train_times, train_times_rev, train_times_nogap, train_missing_x, train_missing_mask), (
                dev_x, dev_y, dev_x_len, dev_x_mask, dev_times, dev_times_rev, dev_times_nogap, dev_missing_x, dev_missing_mask), (
                test_x, test_y, test_x_len, test_x_mask, test_times, test_times_rev, test_times_nogap, test_missing_x, test_missing_mask)
        else:
            return (train_x, train_y, train_x_len, train_x_mask, train_times, train_times_rev, train_missing_x, train_missing_mask), (
                dev_x, dev_y, dev_x_len, dev_x_mask, dev_times, dev_times_rev, dev_missing_x, dev_missing_mask), (
                test_x, test_y, test_x_len, test_x_mask, test_times, test_times_rev, test_missing_x, test_missing_mask)
    else:
        return (train_x, train_y, train_x_len, train_x_mask, train_times, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_x_mask, dev_times, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_x_mask, test_times, test_missing_x, test_missing_mask)