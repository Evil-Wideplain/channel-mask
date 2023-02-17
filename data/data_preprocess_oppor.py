# encoding=utf-8

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle as cp
from pandas import Series
import zipfile
import argparse
from io import BytesIO
from torchvision import transforms
from data.data_preprocess_utils import *
from sklearn.model_selection import StratifiedShuffleSplit

from utils.Util import save_record, args_contains

label = "gestures"
NUM_FEATURES = 77
NUM_TIMESTEPS = 30
NUM_CLASS = 18

"""
"modes of locomotion" classes; low-level actions relating 13 actions to 23 objects; 
17 mid-level gesture classes; 
and 5 high-level activity classes
"""

# https://github.com/STRCWearlab/DeepConvLSTM/blob/master/preprocess_data.py

class data_loader_oppor(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


def load_domain_data(domain_idx):
    """ to load all the data from the specific domain
    :param domain_idx:
    :return: X and y data of the entire domain
    """
    data_dir = DataDir + '/OpportunityUCIDataset/'
    saved_filename = 'oppo_domain_' + domain_idx + '_wd.data'  # with domain label
    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = DataDir + '/OpportunityUCIDataset/dataset/'
        OPPOR_DATA_FILES = [
            'S1-Drill.dat',
            'S1-ADL1.dat',
            'S1-ADL2.dat',
            'S1-ADL3.dat',
            'S1-ADL4.dat',
            'S1-ADL5.dat',

            'S2-Drill.dat',
            'S2-ADL1.dat',
            'S2-ADL2.dat',
            'S2-ADL3.dat',
            'S2-ADL4.dat',
            'S2-ADL5.dat',

            'S3-Drill.dat',
            'S3-ADL1.dat',
            'S3-ADL2.dat',
            'S3-ADL3.dat',
            'S3-ADL4.dat',
            'S3-ADL5.dat',

            'S4-Drill.dat',
            'S4-ADL1.dat',
            'S4-ADL2.dat',
            'S4-ADL3.dat',
            'S4-ADL4.dat',
            'S4-ADL5.dat'
        ]

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        cur_domain_files = [str_folder + a for a in OPPOR_DATA_FILES if a[:2] == domain_idx]

        X, y = load_data_files(label, cur_domain_files)
        print(y)
        # chnge the domain index from string S1 to 0
        d = np.full(y.shape, int(domain_idx[-1]) - 1, dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        # file function is not supported in python3, use open instead
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d


def load_data_files(label, data_files):
    """Loads specified data files' features (x) and labels (y)

    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    :param data_files: list of strings
        Data files to load.
    :return: numpy integer matrix, numy integer array
        Loaded sensor data, segmented into features (x) and labels (y)
    """
    data_x = np.empty((0, NUM_FEATURES))
    data_y = np.empty((0))

    for filename in data_files:
        try:
            data = np.loadtxt(filename)
            print('... file {0} shape:{1}'.format(filename, data.shape))
            # save_record(str(data[23, :]), './xieqi/data.txt')
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))

    return data_x, data_y


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """
    # Select correct columns
    data = select_columns_opp(data)
    # print("after select shape:{}".format(data.shape))
    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    # print(data_x.shape, data_y.shape, data_y)
    # TODO: 输出标签为0 ？？
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)
    # Perform linear interpolation (a.k.a. filling in NaN)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x)

    return data_x, data_y


def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: tuple((numpy integer 2D matrix, numpy integer 1D matrix))
        (Selection of features (N, f), feature_is_accelerometer (f,) one-hot)
    """
    # In term of column_names.txt's ranges: excluded-included (here 0-indexed)
    features_delete = np.arange(0, 37)
    features_delete = np.concatenate([features_delete, np.arange(46, 50)])
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    # Deleting some signals to keep only the 113 of the challenge
    data = np.delete(data, features_delete, 1)
    return data


def divide_x_y(data, label):
    """Segments each sample into (time+features) and (label)

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 0:NUM_FEATURES]
    # Choose labels type for y
    if label not in ['locomotion', 'gestures']:
        raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, NUM_FEATURES]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, (NUM_FEATURES + 1)]  # Gestures label
    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def prep_domains_oppo(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['S1', 'S2', 'S3', 'S4']
    # source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    source_loaders = []
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        print(x_win.shape, y_win.shape, d_win.shape)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_oppor(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_oppor(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_oppor(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_domains_oppo_random_fine(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['S1', 'S2', 'S3', 'S4']
    # source_domain_list.remove(args.target_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # source domain data prep
    source_loaders = []
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        print(x_win.shape, y_win.shape, d_win.shape)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_fine_split(x_win_all, y_win_all, d_win_all,
                                                                   split_ratio=args.ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_oppor(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_oppor(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_oppor(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_oppo(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_domains_oppo(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_oppo_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
