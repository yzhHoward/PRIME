from copy import deepcopy
import pickle
import random


def load_challenge_2012(time_gap = True, ratio = 1):

    x, y, static, mask, name = pickle.load(open('./data/Challenge2012/data.pkl', 'rb'))
    x_len = [len(i) for i in x]
    max_len = max(x_len)

    missing_x = deepcopy(x)
    missing_mask = deepcopy(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k] != 0:
                    if random.random() > ratio:
                        missing_x[i][j][k] = 0
                        missing_mask[i][j][k] = 0
                    else:
                        mask[i][j][k] = 0

    time = []
    time_rev = []
    if time_gap:
        for i in range(len(missing_mask)):
            time_person = []
            time_person_rev = []
            for j in range(len(missing_mask[i])):
                time_feature = []
                time_feature_rev = []
                gap = 0
                gap_rev = 0
                length = len(missing_mask[i][j])
                for k in range(length):
                    time_feature.append(gap)
                    time_feature_rev.append(gap_rev)
                    if missing_mask[i][j][k] == 1:
                        gap = 1
                    else:
                        gap += 1
                    if missing_mask[i][j][length - k - 1] == 1:
                        gap_rev = 1
                    else:
                        gap_rev += 1
                time_person.append(time_feature)
                time_person_rev.append(time_feature_rev[::-1])
            time.append(time_person)
            time_rev.append(time_person_rev)
    else:
        for i in range(len(x_len)):
            # t0 = 1 / x_len[i]
            t0 = 1 / max_len
            time.append([_*t0 for _ in range(x_len[i])])

    train_num = int(len(x) * 0.8) + 1
    dev_num = int(len(x) * 0.1) + 1
    test_num = int(len(x) * 0.1)
    assert (train_num + dev_num + test_num == len(x))

    train_x = []
    train_y = []
    train_static = []
    train_x_len = []
    train_x_mask = []
    train_times = []
    train_times_rev = []
    train_missing_x = []
    train_missing_mask = []
    if time_gap:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_times_rev.append(time_rev[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num):
            train_x.append(x[idx])
            train_y.append(int(y[idx]))
            train_static.append(static[idx])
            train_x_len.append(x_len[idx])
            train_x_mask.append(mask[idx])
            train_times.append(time[idx])
            train_missing_x.append(missing_x[idx])
            train_missing_mask.append(missing_mask[idx])

    dev_x = []
    dev_y = []
    dev_static = []
    dev_x_len = []
    dev_x_mask = []
    dev_times = []
    dev_times_rev = []
    dev_missing_x = []
    dev_missing_mask = []
    if time_gap:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_times_rev.append(time_rev[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num, train_num + dev_num):
            dev_x.append(x[idx])
            dev_y.append(int(y[idx]))
            dev_static.append(static[idx])
            dev_x_len.append(x_len[idx])
            dev_x_mask.append(mask[idx])
            dev_times.append(time[idx])
            dev_missing_x.append(missing_x[idx])
            dev_missing_mask.append(missing_mask[idx])

    test_x = []
    test_y = []
    test_static = []
    test_x_len = []
    test_x_mask = []
    test_times = []
    test_times_rev = []
    test_missing_x = []
    test_missing_mask = []
    if time_gap:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_times_rev.append(time_rev[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])
    else:
        for idx in range(train_num + dev_num, train_num + dev_num + test_num):
            test_x.append(x[idx])
            test_y.append(int(y[idx]))
            test_static.append(static[idx])
            test_x_len.append(x_len[idx])
            test_x_mask.append(mask[idx])
            test_times.append(time[idx])
            test_missing_x.append(missing_x[idx])
            test_missing_mask.append(missing_mask[idx])

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    if time_gap:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_times_rev, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_times_rev, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_times_rev, test_missing_x, test_missing_mask)
    else:
        return (train_x, train_y, train_x_len, train_static, train_x_mask, train_times, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_static, dev_x_mask, dev_times, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_static, test_x_mask, test_times, test_missing_x, test_missing_mask)