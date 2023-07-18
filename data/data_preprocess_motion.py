'''
Data Pre-processing on UCIHAR dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as pk
from data.data_preprocess_utils import get_sample_weights, train_test_val_split, DataDir, train_test_val_fine_split
from data.base_loader import base_loader
import pandas as pd

from utils.Util import args_contains


def get_ds_infos(path):
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes
    """

    dss = pd.read_csv(path, sep=",")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.

    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t + ".x", t + ".y", t + ".z"])
        else:
            dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])

    return dt_list


class data_loader_motion(base_loader):
    def __init__(self, samples, labels, domains, t):
        super(data_loader_motion, self).__init__(samples, labels, domains)
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = self.T(sample)
        return sample, target, domain


def creat_time_series(dt_list, act_labels, trial_codes, labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        combine_grav_acc: True, means adding each axis of gravity to  corresponding axis of userAcceleration.
    Returns:
        It returns a time-series of sensor data.

    """
    num_data_cols = len(dt_list * 3)
    ds_list = get_ds_infos(DataDir + '/DeviceMotion/data_subjects_info.csv')
    print("[INFO] -- Creating Time-Series")
    cols = []
    for axes in dt_list:
        cols += axes
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    dataset_x, dataset_y, dataset_d = None, None, None
    for sub_id in ds_list["code"]:
        if os.path.isfile(DataDir + '/DeviceMotion/domain/sub_{}.data'.format(sub_id)):
            data = np.load(
                DataDir + "/DeviceMotion/domain/sub_{}.data".format(sub_id), allow_pickle=True)
            X = data[0][0]
            Y = data[0][1]
            D = data[0][2]
        else:
            X, Y, D = None, None, None
            for act_id, act in enumerate(act_labels):
                if os.path.isfile(DataDir + '/DeviceMotion/domain_act/sub_{}_act_{}_domain.csv'.format(sub_id, act)):
                    dataset = pd.read_csv(DataDir + "/DeviceMotion/domain_act/sub_{}_act_{}_domain.csv".format(
                        sub_id, act))
                else:
                    if labeled:
                        # "7" --> [act, code, weight, height, age, gender, trial]
                        dataset = np.zeros((0, num_data_cols + 7))
                    else:
                        dataset = np.zeros((0, num_data_cols))

                    for trial in trial_codes[act]:
                        fname = DataDir + '/DeviceMotion/' + act + '_' + str(trial) + '/sub_'+str(int(sub_id))+'.csv'
                        raw_data = pd.read_csv(fname)
                        raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                        vals = np.zeros((len(raw_data), num_data_cols))

                        for x_id, axes in enumerate(dt_list):
                            vals[:, x_id * 3:(x_id + 1) * 3] = raw_data[axes].values
                            vals = vals[:, :num_data_cols]
                        if labeled:
                            lbls = np.array([[act_id,
                                              sub_id - 1,
                                              ds_list["weight"][sub_id - 1],
                                              ds_list["height"][sub_id - 1],
                                              ds_list["age"][sub_id - 1],
                                              ds_list["gender"][sub_id - 1],
                                              trial
                                              ]] * len(raw_data))
                            vals = np.concatenate((vals, lbls), axis=1)
                        dataset = np.append(dataset, vals, axis=0)

                    dataset = pd.DataFrame(data=dataset, columns=cols)
                    dataset.to_csv(
                        DataDir + '/DeviceMotion/domain_act/sub_{}_act_{}_domain.csv'.format(sub_id, act), index=False)

                x = dataset.iloc[:, 0:-7].to_numpy()
                # y = dataset['act'].to_numpy()
                # d = dataset['id'].to_numpy()
                len_x = x.shape[0] // 200
                x = np.concatenate(
                    (x[:len_x * 200, :].reshape(-1, 200, 12), x[-200:, :].reshape(-1, 200, 12)), axis=0)
                # y = np.concatenate(
                #     (y[:len_x * 200, ].reshape(-1, 200), y[-200:, ].reshape(-1, 200)), axis=0)
                # d = np.concatenate(
                #     (d[:len_x * 200, ].reshape(-1, 200), d[-200:, ].reshape(-1, 200)), axis=0)
                y = np.zeros(shape=(x.shape[0]))
                y[:] = act_id
                d = np.zeros(shape=(x.shape[0]))
                d[:] = sub_id
                if X is None:
                    X = x
                else:
                    X = np.concatenate((X, x), axis=0)

                if Y is None:
                    Y = y
                else:
                    Y = np.concatenate((Y, y), axis=0)

                if D is None:
                    D = d
                else:
                    D = np.concatenate((D, d), axis=0)

            print('X:{},Y:{},D:{}'.format(X.shape, Y.shape, D.shape))
            f = open(DataDir + '/DeviceMotion/domain/sub_{}.data'.format(sub_id), 'wb')
            pk.dump([(X, Y, D)], f)
            f.close()

        if dataset_x is None:
            dataset_x = X
        else:
            dataset_x = np.concatenate((dataset_x, X), axis=0)

        if dataset_y is None:
            dataset_y = Y
        else:
            dataset_y = np.concatenate((dataset_y, Y), axis=0)

        if dataset_d is None:
            dataset_d = D
        else:
            dataset_d = np.concatenate((dataset_d, D), axis=0)

    return dataset_x, dataset_y, dataset_d


def prep_domains_motion_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    pass


def prep_domains_motion_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    pass


def prep_domains_motion_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    dt_list = set_data_types(
        data_types=['attitude', 'gravity', 'rotationRate', 'userAcceleration'])
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    num_features = 12
    num_act_labels = 6  # dws, ups, wlk, jog, sit, std
    num_gen_labels = 1  # 0/1(female/male)
    act_labels = ['dws', 'ups', 'wlk', 'jog', 'sit', 'std']
    trial_codes = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [
        7, 8, 15], "jog": [9, 16], "sit": [5, 13], "std": [6, 14]}

    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                          '17', '18', '19', '20', '21', '22', '23', '24']

    x_win_all, y_win_all, d_win_all = creat_time_series(dt_list, act_labels, trial_codes, labeled=True)
    n_train, n_test, ratio = [], 0, 0.0

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
    # print(type(sample_weights))
    # print(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set_r = data_loader_motion(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_motion(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_motion(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_domains_motion_random_fine(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    dt_list = set_data_types(
        data_types=['attitude', 'gravity', 'rotationRate', 'userAcceleration'])
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)
    # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    num_features = 12
    num_act_labels = 6  # dws, ups, wlk, jog, sit, std
    num_gen_labels = 1  # 0/1(female/male)
    act_labels = ['dws', 'ups', 'wlk', 'jog', 'sit', 'std']
    trial_codes = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [
        7, 8, 15], "jog": [9, 16], "sit": [5, 13], "std": [6, 14]}

    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                          '17', '18', '19', '20', '21', '22', '23', '24']

    x_win_all, y_win_all, d_win_all = creat_time_series(dt_list, act_labels, trial_codes, labeled=True)
    n_train, n_test, ratio = [], 0, 0.0

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_fine_split(x_win_all, y_win_all, d_win_all,
                                                                   split_ratio=args.ratio)

    print(x_win_train.shape, y_win_train.shape, d_win_train.shape)
    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    # print(type(sample_weights))
    # print(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set_r = data_loader_motion(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_motion(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_motion(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_motion(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_domains_motion_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject':
        return prep_domains_motion_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_motion_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_motion_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
