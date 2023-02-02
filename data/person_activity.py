import os
import numpy as np
from copy import deepcopy
import torch
from torchvision.datasets.utils import download_url


class PersonActivity(object):
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
    ]

    tag_ids = [
        "010-000-024-033", #"ANKLE_LEFT",
        "010-000-030-096", #"ANKLE_RIGHT",
        "020-000-033-111", #"CHEST",
        "020-000-032-221" #"BELT"
    ]
    
    tag_dict = {k: i for i, k in enumerate(tag_ids)}

    label_names = [
         "walking",
         "falling",
         "lying down",
         "lying",
         "sitting down",
         "sitting",
         "standing up from lying",
         "on all fours",
         "sitting on the ground",
         "standing up from sitting",
         "standing up from sit on grnd"
    ]

    #label_dict = {k: i for i, k in enumerate(label_names)}

    #Merge similar labels into one class
    label_dict = {
        "walking": 0,
         "falling": 1,
         "lying": 2,
         "lying down": 2,
         "sitting": 3,
         "sitting down" : 3,
         "standing up from lying": 4,
         "standing up from sitting": 4,
         "standing up from sit on grnd": 4,
         "on all fours": 5,
         "sitting on the ground": 6
         }


    def __init__(self, root, download=False,
        reduce='average', max_seq_length = 50,
        n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.reduce = reduce
        self.max_seq_length = max_seq_length

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        def save_record(records, record_id, tt, vals, mask, labels):
            tt = torch.tensor(tt).to(self.device)

            vals = torch.stack(vals)
            mask = torch.stack(mask)
            labels = torch.stack(labels)

            # flatten the measurements for different tags
            vals = vals.reshape(vals.size(0), -1)
            mask = mask.reshape(mask.size(0), -1)
            assert(len(tt) == vals.size(0))
            assert(mask.size(0) == vals.size(0))
            assert(labels.size(0) == vals.size(0))

            #records.append((record_id, tt, vals, mask, labels))

            seq_length = len(tt)
            # split the long time series into smaller ones
            offset = 0
            slide = self.max_seq_length // 2

            while (offset + self.max_seq_length < seq_length):
                idx = range(offset, offset + self.max_seq_length)
    
                first_tp = tt[idx][0]
                records.append((record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx]))
                offset += slide

        for url in self.urls:
            filename = url.split('/')[-1]
            # download_url(url, self.raw_folder, filename, None)

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder)
            records = []
            first_tp = None

            with open(os.path.join(dirname, filename)) as f:
                lines = f.readlines()
                prev_time = -1
                tt = []

                record_id = None
                for l in lines:
                    cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
                    value_vec = torch.Tensor((float(val1), float(val2), float(val3))).to(self.device)
                    time = float(time)

                    if cur_record_id != record_id:
                        if record_id is not None:
                            save_record(records, record_id, tt, vals, mask, labels)
                        tt, vals, mask, nobs, labels = [], [], [], [], []
                        record_id = cur_record_id
                    
                        tt = [torch.zeros(1).to(self.device)]
                        vals = [torch.zeros(len(self.tag_ids),3).to(self.device)]
                        mask = [torch.zeros(len(self.tag_ids),3).to(self.device)]
                        nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
                        labels = [torch.zeros(len(self.label_names)).to(self.device)]
                        
                        first_tp = time
                        time = round((time - first_tp)/ 10**5)
                        prev_time = time
                    else:
                        # for speed -- we actually don't need to quantize it in Latent ODE
                        time = round((time - first_tp)/ 10**5) # quatizing by 100 ms. 10,000 is one millisecond, 10,000,000 is one second

                    if time != prev_time:
                        tt.append(time)
                        vals.append(torch.zeros(len(self.tag_ids),3).to(self.device))
                        mask.append(torch.zeros(len(self.tag_ids),3).to(self.device))
                        nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
                        labels.append(torch.zeros(len(self.label_names)).to(self.device))
                        prev_time = time

                    if tag_id in self.tag_ids:
                        n_observations = nobs[-1][self.tag_dict[tag_id]]
                        if (self.reduce == 'average') and (n_observations > 0):
                            prev_val = vals[-1][self.tag_dict[tag_id]]
                            new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
                            vals[-1][self.tag_dict[tag_id]] = new_val
                        else:
                            vals[-1][self.tag_dict[tag_id]] = value_vec

                        mask[-1][self.tag_dict[tag_id]] = 1
                        nobs[-1][self.tag_dict[tag_id]] += 1

                        if label in self.label_names:
                            if torch.sum(labels[-1][self.label_dict[label]]) == 0:
                                labels[-1][self.label_dict[label]] = 1
                    else:
                        assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
                save_record(records, record_id, tt, vals, mask, labels)
            
            torch.save(
                records,
                os.path.join(self.processed_folder, 'data.pt')
            )
                
        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]
            if not os.path.exists(
                os.path.join(self.processed_folder, 'data.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root)

    @property
    def processed_folder(self):
        return os.path.join(self.root)

    @property
    def data_file(self):
        return 'data.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Max length: {}\n'.format(self.max_seq_length)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str


def get_person_id(record_id):
    # The first letter is the person id
    person_id = record_id[0]
    person_id = ord(person_id) - ord("A")
    return person_id


def load_person_activity(time_gap = True, ratio = 1):
    dataset_obj = PersonActivity('data/PersonActivity', download=True)
    data = dataset_obj.data
    name = [d[0] for d in data]
    t = torch.stack([d[1] for d in data]).float().numpy()
    x = torch.stack([d[2] for d in data]).numpy()
    mask = torch.stack([d[3] for d in data]).numpy()
    y = torch.stack([d[4] for d in data]).numpy()
    x_len = np.full((x.shape[0],), x.shape[1])
    x_masked = np.ma.masked_array(x.reshape(-1, x.shape[2]), mask.reshape(-1, x.shape[2])==0)
    x_mean = np.mean(x_masked, 0)
    x_std = np.std(x_masked, 0)
    x = (x - x_mean) / x_std

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

    dev_x = x[index[train_num:train_num+dev_num]]
    dev_y = y[index[train_num:train_num+dev_num]]
    dev_x_len = x_len[index[train_num:train_num+dev_num]]
    dev_x_mask = mask[index[train_num:train_num+dev_num]]
    dev_times = time[index[train_num:train_num+dev_num]]
    if time_gap:
        dev_times_rev = time_rev[index[train_num:train_num+dev_num]]
    dev_missing_x = missing_x[index[train_num:train_num+dev_num]]
    dev_missing_mask = missing_mask[index[train_num:train_num+dev_num]]

    test_x = x[index[-test_num:]]
    test_y = y[index[-test_num:]]
    test_x_len = x_len[index[-test_num:]]
    test_x_mask = mask[index[-test_num:]]
    test_times = time[index[-test_num:]]
    if time_gap:
        test_times_rev = time_rev[index[-test_num:]]
    test_missing_x = missing_x[index[-test_num:]]
    test_missing_mask = missing_mask[index[-test_num:]]

    assert (len(train_x) == train_num)
    assert (len(dev_x) == dev_num)
    assert (len(test_x) == test_num)
    if time_gap:
        return (train_x, train_y, train_x_len, train_x_mask, train_times, train_times_rev, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_x_mask, dev_times, dev_times_rev, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_x_mask, test_times, test_times_rev, test_missing_x, test_missing_mask)
    else:
        return (train_x, train_y, train_x_len, train_x_mask, train_times, train_missing_x, train_missing_mask), (
            dev_x, dev_y, dev_x_len, dev_x_mask, dev_times, dev_missing_x, dev_missing_mask), (
            test_x, test_y, test_x_len, test_x_mask, test_times, test_missing_x, test_missing_mask)
