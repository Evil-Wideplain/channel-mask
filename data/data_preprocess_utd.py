"""
https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
Data Pre-processing on UTD-MHAD.

UTD-MHAD dataset consists of 27 different actions: (1) right arm swipe to the left, (2) right arm swipe to the right,
(3) right hand wave, (4) two hand front clap, (5) right arm throw, (6) cross arms in the chest, (7) basketball shoot,
(8) right hand draw x, (9) right hand draw circle (clockwise), (10) right hand draw circle (counter clockwise),
(11) draw triangle, (12) bowling (right hand), (13) front boxing, (14) baseball swing from right, (15) tennis right
hand forehand swing, (16) arm curl (two arms), (17) tennis serve, (18) two hand push, (19) right hand knock on door,
(20) right hand catch an object, (21) right hand pick up and throw, (22) jogging in place, (23) walking in place,
(24) sit to stand, (25) stand to sit, (26) forward lunge (left foot forward), (27) squat (two arms stretch out).

performed by 8 subjects (4 females and 4 males).

Each subject repeated each action 4 times.

After removing three corrupted sequences, the dataset includes 861 data sequences. Four data modalities of RGB videos,
depth videos, skeleton joint positions, and the inertial sensor signals were recorded in three channels or threads.
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle as cp
from data.data_preprocess_utils import *
import scipy.io
from data.base_loader import base_loader
from utils.Util import args_contains

n_classes = 27
n_sub = 8
n_trial = 4


def load_domain_data(domain_idx, SLIDING_WINDOW_LEN=100, SLIDING_WINDOW_STEP=50):
    """ to load all the data from the specific domain with index domain_idx
    :param SLIDING_WINDOW_STEP: split timesteps 时间步长 窗口长度
    :param SLIDING_WINDOW_LEN: 窗口步长
    :param domain_idx: index of a single domain
    :return: X and y data of the entire domain
    """
    data_dir = DataDir + '/UTD-MHAD/'
    saved_filename = 'utd_domain_' + domain_idx + '_wd.data'  # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        x_all = data[0][0]
        y_all = data[0][1]
        d_all = data[0][2]
    else:
        str_folder = DataDir + '/UTD-MHAD/Inertial/'
        x_all, y_all, d_all = None, None, None
        for m in range(n_classes):
            _m = m + 1
            for n in range(n_trial):
                _n = n + 1
                mat_file = "a{}_s{}_t{}_inertial.mat".format(_m, domain_idx, _n)
                if os.path.exists(str_folder + mat_file):
                    data = scipy.io.loadmat(str_folder + mat_file)
                    X = data['d_iner']
                    Y = np.full(X.shape[0], m, dtype=int)
                    D = np.full(X.shape[0], domain_idx, dtype=int)
                    X, Y, D = opp_sliding_window_w_d(X, Y, D, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
                    print(X.shape, Y.shape, D.shape)
                    x_all = np.concatenate(
                        (x_all, X), axis=0) if x_all is not None else X
                    y_all = np.concatenate(
                        (y_all, Y), axis=0) if y_all is not None else Y
                    d_all = np.concatenate(
                        (d_all, D), axis=0) if d_all is not None else D
        x_all, y_all, d_all = np.array(x_all), np.array(y_all), np.array(d_all)

        print(x_all.shape, y_all.shape, d_all.shape)
        obj = [(x_all, y_all, d_all)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return x_all, y_all, d_all


class data_loader_utd(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_utd, self).__init__(samples, labels, domains)


def prep_domains_utd_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
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

    data_set = data_loader_utd(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                               num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    unique_y, counts_y = np.unique(y, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)

    data_set = data_loader_utd(x, y, d)
    # shuffle is forced to be False when sampler is available
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader


def prep_domains_utd_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        # todo: index change of domain ID is different from smaller indices; can be combined to a function when time is more available
        x, y, d = load_domain_data(source_domain, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

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

    data_set = data_loader_utd(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                               num_workers=workers, pin_memory=pin_memory)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_utd(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader


def prep_domains_utd_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_utd(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_utd(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_utd(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_domains_utd_random_fine(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_utd(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_utd(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_utd(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_utd(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'subject':
        return prep_domains_utd_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_utd_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random':
        return prep_domains_utd_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_utd_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error!\n'
