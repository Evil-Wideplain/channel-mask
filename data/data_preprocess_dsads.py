"""
Data Pre-processing on Daily and Sports Activities Data Set.
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from data.data_preprocess_utils import get_sample_weights, train_test_val_split, DataDir, train_test_val_fine_split
from data.base_loader import base_loader
from utils.Util import args_contains

n_classes = 19
n_sub = 8
n_trial = 60


def format_data(datafile):
    data = np.loadtxt(datafile, dtype=np.float32)
    return data


def format_path_num(num):
    if type(num) == str:
        return num
    if num < 10:
        return "0{}".format(num)
    else:
        return "{}".format(num)


def load_domain_data(domain_idx):
    """ to load all the data from the specific domain with index domain_idx
    :param domain_idx: index of a single domain
    :return: X and y data of the entire domain
    """
    data_dir = DataDir + '/DSADS/'
    saved_filename = 'dsads_domain_' + domain_idx + '_wd.data'  # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        str_folder = DataDir + '/DSADS/data/'
        x_all, y_all, d_all = [], [], []
        for m in range(n_classes):
            _m = format_path_num(m + 1)
            for n in range(n_trial):
                _n = format_path_num(n + 1)
                txt_file = "a{}/p{}/s{}.txt".format(_m, domain_idx, _n)
                data = np.loadtxt(str_folder + txt_file, delimiter=",", dtype=np.float32)
                assert data.shape == (125, 45)

                x_all.append(data)
                y_all.append(m)
                d_all.append(int(domain_idx) - 1)
        X, y, d = np.array(x_all), np.array(y_all), np.array(d_all)
        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))
    return X, y, d


class data_loader_dsads(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_dsads, self).__init__(samples, labels, domains)


def prep_domains_dsads_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # todo: make the domain IDs as arguments or a function with args to select the IDs (default, customized, small, etc)
    source_domain_list = ['1', '2', '3', '4', '5']

    source_domain_list.remove(args.target_domain)

    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

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

    data_set = data_loader_dsads(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                               num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_dsads(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader


def prep_domains_dsads_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    source_domain_list.remove(args.target_domain)

    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

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

    data_set = data_loader_dsads(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_dsads(x, y, d)
    # todo: the batch size can be different for some ttt models, tbc
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    print('target_loader batch: ', len(target_loader))

    return source_loaders, None, target_loader


def prep_domains_dsads_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)

    train_set_r = data_loader_dsads(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_dsads(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_dsads(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_domains_dsads_random_fine(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0

    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    print("x_win_all: {}, y_win_all: {}, d_win_all: {}".format(x_win_all.shape, y_win_all.shape, d_win_all.shape))
    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_fine_split(x_win_all, y_win_all, d_win_all,
                                                                   split_ratio=args.ratio)

    print("x_win_train: {}, x_win_val: {}, x_win_test: {}, y_win_train: {}, y_win_val: {}, y_win_test: {}, "
          "d_win_train: {}, d_win_val: {}, d_win_test: {}"
          .format(x_win_train.shape, x_win_val.shape, x_win_test.shape, y_win_train.shape, y_win_val.shape,
                  y_win_test.shape, d_win_train.shape, d_win_val.shape, d_win_test.shape))

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_dsads(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_dsads(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_dsads(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False, num_workers=workers,
                               pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_dsads(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_domains_dsads_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject':
        return prep_domains_dsads_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_dsads_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_dsads_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
