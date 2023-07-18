"""
Data Pre-processing on WISDM dataset.
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from data.data_preprocess_utils import *
from data.base_loader import base_loader
import pandas as pd

from utils.Util import args_contains


def format_data(datafile):
    columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(datafile, header=None, names=columns)
    df = df.dropna(axis=0)
    return df.to_numpy()


def adjust_idx_labels(y):
    y_new = np.zeros_like(y, dtype=int)
    y_new[y == 'Downstairs'] = 0
    y_new[y == 'Jogging'] = 1
    y_new[y == 'Sitting'] = 2
    y_new[y == 'Standing'] = 3
    y_new[y == 'Upstairs'] = 4
    y_new[y == 'Walking'] = 5
    return y_new


def load_domain_data(domain_idx):
    """ to load all the data from the specific domain with index domain_idx
    :param domain_idx: index of a single domain
    :return: X and y data of the entire domain
    """
    data_dir = DataDir + '/WISDM/'
    saved_filename = 'wisdm_domain_' + domain_idx + '_wd.data'  # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        Y = data[0][1]
        D = data[0][2]
    else:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        str_file = DataDir + '/WISDM/WISDM_ar_v1.1_raw.txt'
        INPUT_SIGNAL_TYPES = [
            "x-axis",
            "y-axis",
            "z-axis",
        ]
        data = format_data(str_file)
        id_all = data[:, 0]
        # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        # 29, 30, 31, 32, 33, 34, 35, 36
        Y_all = data[:, 1]
        X_all = data[:, 3:]
        X_all[:, 2] = np.array([x.replace(';', '') for x in X_all[:, 2]], dtype=np.float)
        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        Y = Y_all[target_idx]
        Y = adjust_idx_labels(Y)
        D = np.full(Y.shape, int(domain_idx), dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, Y.shape, D.shape))

        obj = [(X, Y, D)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, Y, D


class data_loader_wisdm(base_loader):
    def __init__(self, samples, labels, domains, t):
        super(data_loader_wisdm, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain


def prep_domains_wisdm_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # todo: make the domain IDs as arguments or a function with args to select the IDs (default, customized, small, etc)
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                          '33', '34', '35', '36']
    source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        # n_channel should be 3, H: 1, W:90
        x, y, d = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        # the  WISDM dataset is segmented by sliding window as default
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)
    transform = None

    data_set = data_loader_wisdm(x_win_all, y_win_all, d_win_all, transform)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    # x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    x, y, d = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_wisdm(x, y, d, transform)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader


def prep_domains_wisdm_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                          '33', '34', '35', '36']
    source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        # n_channel should be 3, H: 1, W:90
        # x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
        x, y, d = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        # the WISDM dataset is segmented by sliding window as default
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)
    transform = None

    data_set = data_loader_wisdm(x_win_all, y_win_all, d_win_all, transform)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    # x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    x, y, d = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_wisdm(x, y, d, transform)
    # todo: the batch size can be different for some ttt models, tbc
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    print('target_loader batch: ', len(target_loader))

    return source_loaders, None, target_loader


def prep_domains_wisdm_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                          '33', '34', '35', '36']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x_win, y_win, d_win = load_domain_data(source_domain)

        # n_channel should be 3, H: 1, W:90
        # x_win = np.transpose(x_win.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
        x_win, y_win, d_win = opp_sliding_window_w_d(x_win, y_win, d_win, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        # print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    print("x_win_all: {}, y_win_all: {}, d_win_all: {}".format(x_win_all.shape, y_win_all.shape, d_win_all.shape))
    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.ratio)

    print("x_win_train: {}, x_win_val: {}, x_win_test: {}, y_win_train: {}, y_win_val: {}, y_win_test: {}, "
          "d_win_train: {}, d_win_val: {}, d_win_test: {}"
          .format(x_win_train.shape, x_win_val.shape, x_win_test.shape, y_win_train.shape, y_win_val.shape,
                  y_win_test.shape, d_win_train.shape, d_win_val.shape, d_win_test.shape))
    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    transform = None
    train_set_r = data_loader_wisdm(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_wisdm(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_wisdm(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_domains_wisdm_random_fine(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                          '33', '34', '35', '36']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x_win, y_win, d_win = load_domain_data(source_domain)

        # n_channel should be 3, H: 1, W:90
        # x_win = np.transpose(x_win.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
        x_win, y_win, d_win = opp_sliding_window_w_d(x_win, y_win, d_win, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        # print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_fine_split(x_win_all, y_win_all, d_win_all,
                                                                   split_ratio=args.ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    transform = None
    train_set_r = data_loader_wisdm(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_wisdm(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_wisdm(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_wisdm(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_domains_wisdm_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject':
        return prep_domains_wisdm_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_wisdm_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_wisdm_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
