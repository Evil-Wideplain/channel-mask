import os
import torch
from torchvision import datasets, transforms
# import torchvision.transforms.functional as FT
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import PIL
import cv2
import pickle
import warnings
import math

from data import data_preprocess_ucihar, data_preprocess_shar, data_preprocess_hhar, data_preprocess_motion, \
    data_preprocess_oppor, data_preprocess_pamap2, data_preprocess_wisdm, data_preprocess_usc, data_preprocess_dsads, \
    data_preprocess_utd
from data.base_loader import HARDataset
from utils.Util import args_contains
from aug.HAR import *
from aug.IMG import *


class HARLoader():
    def __init__(self, args):
        self.args = args
        self.path = args_contains(self.args, 'data_dir', '..')
        # self.n_timestep, self.n_features, self.outputs = None, None, None
        # self.data_shape = None
        # self.trainset, self.valset, self.testset = self.train_test_set()

    # 数据增强方法
    def tansform(self):
        multiplier = args_contains(self.args, 'multiplier', 1)
        if multiplier == 1:
            train_transform = None
            # train_transform = transforms.Compose([
                # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1)),
                # transforms.ToTensor(),
            # ])
        else:
            train_transform = transforms.Compose([
                # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1)),
                # Resample(1, 0),
                ResampleRandom(),
                transforms.ToTensor(),
            ])
        # 测试就不需要数据增强了
        test_transform = None

        return train_transform, test_transform

    # 加载数据  默认是cifar imagenet数据集的还有加上去，跑不起来，所以上面的数据增强部分其实用到的也就是只有cifar10
    def train_test_set(self):
        # TODO: 8.数据增强方法
        train_transform, test_transform = self.tansform()

        trainset, valset, testset = None, None, None

        dataset_name = args_contains(self.args, 'dataset', 'UCI')
        ratio = args_contains(self.args, 'ratio', '0.01')
        ratio = float(ratio.split(',')[0])
        seed = args_contains(self.args, 'seed', 888)

        if dataset_name == 'UCI':
            # (10299, 128, 9), (10299, 6)
            dataset_X = np.load(self.path + '/datasets/UCI_X.npy')
            dataset_Y = np.load(self.path + '/datasets/UCI_Y.npy')

            train_x, train_y = dataset_X, dataset_Y
            if ratio == 1.0:
                test_x, val_x, test_y, val_y = dataset_X, dataset_X, dataset_Y, dataset_Y
            else:
                test_x, val_x, test_y, val_y = train_test_split(
                    dataset_X, dataset_Y, test_size=ratio, random_state=seed)

            # self.n_timestep, self.n_features, self.outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
            # self.data_shape = train_x.shape[1:]
            # TODO: 9.转换成Dataloader类型,并且对每一个批量的数据都进行数据增强
            trainset = HARDataset(
                data_tensor=torch.from_numpy(train_x).float(), target_tensor=torch.from_numpy(train_y).float(),
                transforms=train_transform)
            valset = HARDataset(
                data_tensor=torch.from_numpy(val_x).float(), target_tensor=torch.from_numpy(val_y).float(),
                transforms=test_transform)
            testset = HARDataset(
                data_tensor=torch.from_numpy(test_x).float(), target_tensor=torch.from_numpy(test_y).float(),
                transforms=test_transform)
        elif dataset_name == 'USC':
            # (17902, 200, 6), (17902, 12)
            dataset_X = np.load(self.path + '/datasets/USCHAD_X.npy')
            dataset_Y = np.load(self.path + '/datasets/USCHAD_Y.npy')

            train_x, train_y = dataset_X, dataset_Y

            if ratio == 1.0:
                test_x, val_x, test_y, val_y = dataset_X, dataset_X, dataset_Y, dataset_Y
            else:
                test_x, val_x, test_y, val_y = train_test_split(
                    dataset_X, dataset_Y, test_size=ratio, random_state=seed)

            # self.n_timestep, self.n_features, self.outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
            # self.data_shape = train_x.shape[1:]
            trainset = HARDataset(
                data_tensor=torch.from_numpy(train_x), target_tensor=torch.from_numpy(train_y).float(),
                transforms=train_transform)
            valset = HARDataset(
                data_tensor=torch.from_numpy(val_x), target_tensor=torch.from_numpy(val_y), transforms=test_transform)
            testset = HARDataset(
                data_tensor=torch.from_numpy(test_x), target_tensor=torch.from_numpy(test_y), transforms=test_transform)
        elif dataset_name == 'Motion':
            # (13592, 200, 6), (13592, 6)
            dataset_X = np.load(self.path + '/datasets/Motion_X.npy')
            dataset_Y = np.load(self.path + '/datasets/Motion_Y.npy')

            train_x, train_y = dataset_X, dataset_Y

            if ratio == 1.0:
                test_x, val_x, test_y, val_y = dataset_X, dataset_X, dataset_Y, dataset_Y
            else:
                test_x, val_x, test_y, val_y = train_test_split(
                    dataset_X, dataset_Y, test_size=ratio, random_state=seed)

            # self.n_timestep, self.n_features, self.outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
            # self.data_shape = train_x.shape[1:]
            trainset = HARDataset(
                data_tensor=torch.from_numpy(train_x), target_tensor=torch.from_numpy(train_y).float(),
                transforms=train_transform)
            valset = HARDataset(
                data_tensor=torch.from_numpy(val_x), target_tensor=torch.from_numpy(val_y), transforms=test_transform)
            testset = HARDataset(
                data_tensor=torch.from_numpy(test_x), target_tensor=torch.from_numpy(test_y), transforms=test_transform)
        else:
            warnings.warn(message='没有这个数据集', category=ResourceWarning)

        return trainset, valset, testset

    # 把加载的数据集转换成pytorch提供的loader对象 并且使用他提供的方法对数据集进行数据增强
    def data_loader(self):
        """
        训练集 trainset
        微调数据集 valset
        测试集 testset
        :return: 返回train_loader, val_loader, test_loader
        """
        # trainset, valset, testset = self.train_test_set(path=self.path)
        trainset, valset, testset = self.train_test_set()

        # https://blog.csdn.net/SweetWind1996/article/details/105328385
        trainsampler = torch.utils.data.sampler.RandomSampler(trainset, replacement=False)
        valsampler = torch.utils.data.sampler.RandomSampler(valset, replacement=False)
        testsampler = torch.utils.data.sampler.RandomSampler(testset, replacement=False)

        # 所以这里才有明明batch大小是batch_size，但是下面loader里拿出来的是2*batch_size(batch)是因为对每一个都采样了两次

        multiplier = args_contains(self.args, 'multiplier', 1)
        batch_size = args_contains(self.args, 'batch_size', 1024)
        val_batch_size = args_contains(self.args, 'val_batch_size', 50)
        test_batch_size = args_contains(self.args, 'test_batch_size', 2048)

        train_batch_sampler = MultiplyBatchSampler(
            multiplier, trainsampler, batch_size, drop_last=False)

        val_batch_sampler = MultiplyBatchSampler(
            1, valsampler, val_batch_size, drop_last=False)

        test_batch_sampler = MultiplyBatchSampler(
            1, testsampler, test_batch_size, drop_last=False)

        # continue sample
        # 这里要不要去掉 再试试 如果单独的聚类效果还是不行的话
        train_iter = args_contains(self.args, 'train_iter', 0)
        val_iters = args_contains(self.args, 'val_iters', 0)
        test_iters = args_contains(self.args, 'test_iters', 0)

        if train_iter != 0:
            train_batch_sampler = ContinousSampler(
                train_batch_sampler,
                train_iter
            )
        if val_iters != 0:
            val_batch_sampler = ContinousSampler(
                val_batch_sampler,
                val_iters
            )

        if test_iters != 0:
            test_batch_sampler = ContinousSampler(
                test_batch_sampler,
                test_iters
            )

        workers = args_contains(self.args, 'workers', 2)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=train_batch_sampler,
        )

        val_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=val_batch_sampler,
        )

        test_loader = torch.utils.data.DataLoader(
            testset,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=test_batch_sampler,
        )

        return [train_loader], val_loader, test_loader


class MultiplyBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, MULTILPLIER, sampler, batch_size, drop_last=True):
        self.MULTILPLIER = MULTILPLIER
        # self.sampler = sampler
        # self.batch_size = batch_size
        # self.drop_last = drop_last
        if type(batch_size) != int:
            batch_size = batch_size.item()

        super().__init__(sampler, int(batch_size), drop_last)

    def __iter__(self):
        for batch in super().__iter__():
            # print('sampler check',batch, batch*self.MULTILPLIER)
            yield batch * self.MULTILPLIER


class ContinousSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, sampler, n_iterations):
        self.base_sampler = sampler
        self.n_iterations = n_iterations

    def __iter__(self):
        cur_iter = 0
        while cur_iter < self.n_iterations:
            for batch in self.base_sampler:
                yield batch
                cur_iter += 1
                if cur_iter >= self.n_iterations:
                    return

    def __len__(self):
        return self.n_iterations

    def set_epoch(self, epoch):
        self.base_sampler.set_epoch(epoch)


def dataloader(args):
    train_loaders, val_loader, test_loader = None, None, None
    # random 将数据分成三份 train((1-ratio)^2) val((1-ratio)*ratio) test(ratio)
    # random fine 将数据分成三份 train((1-ratio)^2) val((1-ratio)*ratio) test(ratio)
    # subject 输出两个部分 train none test (train test的数据一致)
    PAA_window_size = args_contains(args, 'PAA_window_size', 2)
    toImage = args_contains(args, 'toImage', False)
    if args.dataset == 'ucihar':
        args.n_features = 9
        args.n_timesteps = 128
        args.n_class = 6
        args.shape = (1, 128, 9)
        if args.cases == 'random_six_channels':
            args.n_features = 6
            args.shape = (1, 128, 6)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args,
                                                                                    SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                    SLIDING_WINDOW_STEP=int(
                                                                                        args.n_timesteps * 0.5))
    if args.dataset == 'shar':
        args.n_features = 3
        args.n_timesteps = 151
        args.n_class = 17
        args.shape = (1, 151, 3)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '1'
        train_loaders, val_loader, test_loader = data_preprocess_shar.prep_shar(args,
                                                                                SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                SLIDING_WINDOW_STEP=int(
                                                                                    args.n_timesteps * 0.5))
    if args.dataset == 'hhar':
        args.n_features = 6
        args.n_timesteps = 100
        args.n_class = 6
        args.shape = (1, 100, 6)
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args,
                                                                                SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                SLIDING_WINDOW_STEP=int(
                                                                                    args.n_timesteps * 0.5),
                                                                                # default Phone
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)
    if args.dataset == 'deviceMotion':
        args.n_features = 12
        args.n_timesteps = 200
        args.n_class = 6
        args.shape = (1, 200, 12)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '0'
        train_loaders, val_loader, test_loader = data_preprocess_motion.prep_motion(args,
                                                                                    SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                    SLIDING_WINDOW_STEP=int(
                                                                                        args.n_timesteps * 0.5))
    if args.dataset == 'opportunity':
        args.n_features = 77
        args.n_timesteps = 30
        args.n_class = 18
        args.shape = (1, 30, 77)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = 'S1'
        train_loaders, val_loader, test_loader = data_preprocess_oppor.prep_oppo(args,
                                                                                 SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                 SLIDING_WINDOW_STEP=int(
                                                                                 args.n_timesteps * 0.5))
    if args.dataset == 'pamap2':
        args.n_features = 52
        args.n_timesteps = 170
        args.n_class = 12
        args.shape = (1, 170, 52)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '101'
        train_loaders, val_loader, test_loader = data_preprocess_pamap2.prep_pamap2(args,
                                                                                  SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                  SLIDING_WINDOW_STEP=int(
                                                                                      args.n_timesteps * 0.5))
    if args.dataset == 'WISDM':
        args.n_features = 3
        args.n_timesteps = 90
        args.n_class = 6
        args.shape = (1, 90, 3)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '1'
        train_loaders, val_loader, test_loader = data_preprocess_wisdm.prep_wisdm(args,
                                                                                  SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                  SLIDING_WINDOW_STEP=80)
    if args.dataset == 'USCHAR':
        args.n_features = 6
        args.n_timesteps = 200
        args.n_class = 12
        args.shape = (1, 200, 6)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '1'
        train_loaders, val_loader, test_loader = data_preprocess_usc.prep_usc(args,
                                                                              SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                              SLIDING_WINDOW_STEP=
                                                                              int(args.n_timesteps*0.5))
    if args.dataset == 'DSADS':
        args.n_features = 45
        args.n_timesteps = 125
        args.n_class = 19
        args.shape = (1, 125, 45)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '1'
        train_loaders, val_loader, test_loader = data_preprocess_dsads.prep_dsads(args,
                                                                                  SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                                  SLIDING_WINDOW_STEP=
                                                                                  int(args.n_timesteps*0.5))
    if args.dataset == 'UTD':
        args.n_features = 6
        args.n_timesteps = 100
        args.n_class = 27
        args.shape = (1, 100, 6)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '1'
        train_loaders, val_loader, test_loader = data_preprocess_utd.prep_utd(args,
                                                                              SLIDING_WINDOW_LEN=args.n_timesteps,
                                                                              SLIDING_WINDOW_STEP=
                                                                              int(args.n_timesteps*0.5))

    # if args.dataset == 'UCI':
    #     args.n_features = 9
    #     args.n_timesteps = 128
    #     args.n_class = 6
    #     args.shape = (1, 128, 9)
    #     loader = HARLoader(args)
    #     train_loaders, val_loader, test_loader = loader.data_loader()
    # if args.dataset == 'USC':
    #     args.n_features = 6
    #     args.n_timesteps = 200
    #     args.n_class = 12
    #     args.shape = (1, 200, 6)
    #     loader = HARLoader(args)
    #     train_loaders, val_loader, test_loader = loader.data_loader()
    # if args.dataset == 'Motion':
    #     args.n_features = 6
    #     args.n_timesteps = 200
    #     args.n_class = 6
    #     args.shape = (1, 200, 6)
    #     loader = HARLoader(args)
    #     train_loaders, val_loader, test_loader = loader.data_loader()

    if toImage:
        args.n_width = math.floor((args.n_timesteps + 1) / PAA_window_size)
        args.n_height = math.floor((args.n_timesteps + 1) / PAA_window_size)
        args.shape = (1, args.n_features, args.n_width, args.n_height)

    return args, train_loaders, val_loader, test_loader
