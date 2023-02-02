import math
import numpy as np
import torch
import torch.nn.functional as F


def pad_one(sents, max_length=None):
    sents_padded = []
    if max_length is None:
        max_length = max([len(_) for _ in sents])
    for i in sents:
        padded = list(i) + [0] * (max_length - len(i))
        sents_padded.append(padded)
    return np.array(sents_padded)


def pad_sents(sents, pad_token, max_length=None):
    sents_padded = []
    if max_length is None:
        max_length = max([len(_) for _ in sents])
    for i in sents:
        padded = list(i) + [pad_token] * (max_length - len(i))
        sents_padded.append(np.array(padded))
    return np.array(sents_padded)


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    device = length.device if device is None else device
    mask = torch.arange(max_len,
                        device=device, dtype=length.dtype).expand(
                            len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def batch_iter(args, batch_size=256, shuffle=False, return_index=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(args[0]) / batch_size)  # 向下取整
    index_array = list(range(len(args[0])))
    arg_len = len(args) + 1 if return_index else len(args)

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size:(i + 1) *
                              batch_size]  # fetch out all the induces

        examples = []
        for idx in indices:
            e = [arg[idx] if arg is not None else None for arg in args]
            if return_index:
                e.append(idx)
            examples.append(e)

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        yield [[e[j] for e in examples] for j in range(arg_len)]


def batch_iter_fast(args, batch_size=256, shuffle=False, return_index=False):
    batch_num = math.ceil(len(args[0]) / batch_size)
    if shuffle:
        idx = torch.randperm(len(args[0]))
    else:
        idx = torch.arange(0, len(args[0]), dtype=torch.long)
    for i in range(batch_num):
        batch = idx[i * batch_size:(i + 1) * batch_size]
        ret = [[arg[_] for _ in batch.tolist()] if isinstance(arg, list) else arg[batch] for arg in args]
        if return_index:
            ret.append(batch)
        yield ret
        # yield [arg[idx[i * batch_size:(i + 1) * batch_size]] for arg in args]


def get_loss(y_pred, y_true):
    # weight = y_true * 0.7 + (1 - y_true) * 0.3
    return F.binary_cross_entropy(y_pred, y_true, weight=None)


def get_ce_loss(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


def get_re_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)


def get_data_by_index(idxs, *args):
    ret = []
    for arg in args:
        ret.append([arg[i] for i in idxs])
    return ret


def get_n2n_data(x, y, x_len, *args):
    length = len(x)
    assert length == len(y)
    assert length == len(x_len)
    for arg in args:
        assert length == len(arg)
    arg_len = len(args)
    new_args = []
    for i in range(arg_len + 3):
        new_args.append([])
    for i in range(length):
        assert len(x[i]) == len(y[i])
        for arg in args:
            assert len(x[i]) == len(arg[i])
        for j in range(len(x[i])):
            new_args[0].append(x[i][:j + 1])
            new_args[1].append(y[i][j])
            new_args[2].append(j + 1)
            for k in range(arg_len):
                new_args[k + 3].append(args[k][i][:j + 1])
    return new_args


def split_train_valid_test(dataset, train_ratio=0.8, dev_ratio=0.1):
    """
    test_ratio is calculated by train_ratio and dev_ratio
    """
    all_num = len(dataset[0])
    train_num = int(all_num * train_ratio)
    dev_num = int(all_num * dev_ratio)
    test_num = all_num - train_num - dev_num

    train_set = []
    dev_set = []
    test_set = []
    for data in dataset:
        train_set.append(data[:train_num])
        dev_set.append(data[train_num:train_num + dev_num])
        test_set.append(data[train_num + dev_num:])

    return train_set, dev_set, test_set
