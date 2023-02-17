import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import PIL
import cv2
import pickle
import warnings
from data.base_loader import HARDataset
from utils.Util import args_contains
from aug.IMG import *


class dataLoader():
    def __init__(self, args):
        self.args = args
        self.path = args_contains(self.args, 'datasets_path', './xieqi')
        self.n_timestep, self.n_features, self.outputs = None, None, None
        self.data_shape = None
        self.trainset, self.valset, self.testset = self.train_test_set(path=self.path)

    # 数据增强方法
    def tansform(self):
        # imagenet 数据增强方法
        dataset_name = args_contains(self.args, 'dataset_name', 'UCI')
        if dataset_name == 'imagenet':
            scale_lower = args_contains(self.args, 'scale_lower', 0.08)
            color_dist_s = args_contains(self.args, 'color_dist_s', 0.5)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    32,
                    scale=(scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(s=color_dist_s),
                transforms.ToTensor(),
                Clip(),
            ])
            test_transform = train_transform
        # cifar10 数据集增强方法
        # 随即裁剪 水平翻转 颜色 高斯模糊
        # 归一化和转为Tensor格式
        elif dataset_name == "cifar":
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        else:
            train_transform, test_transform = None, None

        return train_transform, test_transform

    # 加载数据  默认是cifar imagenet数据集的还有加上去，跑不起来，所以上面的数据增强部分其实用到的也就是只有cifar10
    def train_test_set(self, path='./xieqi'):
        # 微调数据集
        def get_valset(train, val_ratio=0.1):
            d_len = len(train)
            _, valset = torch.utils.data.random_split(
                train, [int(d_len * (1 - val_ratio)), int(d_len * val_ratio)])
            train_x = []
            train_y = []
            for i in range(len(valset)):
                train_x.append(valset[i][0].numpy())
                train_y.append(valset[i][1])
            # for i in range(10):
            # print(np.sum(np.array(train_y) == i))

            return valset, np.array(train_x), np.array(train_y)

        def get_valset_2(train, val_ratio=0.1, path=path + '/datasets/cifar10.npy'):
            if not os.path.exists(path):
                train_x = []
                train_y = []
                for i in range(len(train)):
                    train_x.append(train[i][0].numpy())
                    train_y.append(train[i][1])

                with open(path, "wb") as f:
                    pickle.dump({
                        "train_x": np.array(train_x),
                        "train_y": np.array(train_y)
                    }, f)

            with open(path, "rb") as f:
                data = pickle.load(f)

            x, y = data['train_x'], data['train_y']

            _, val_x, _, val_y = train_test_split(
                x, y, test_size=val_ratio, shuffle=True, random_state=88)

            valset = torch.utils.data.TensorDataset(
                torch.from_numpy(val_x), torch.from_numpy(val_y))
            return valset, val_x, val_y

        # TODO: 8.数据增强方法
        train_transform, test_transform = self.tansform()

        trainset, valset, testset = None, None, None

        dataset_name = args_contains(self.args, 'dataset_name', 'UCI')
        val_ratio = args_contains(self.args, 'ratio', '0.01')
        val_ratio = float(val_ratio.split(',')[0])
        seed = args_contains(self.args, 'seed', 888)

        if dataset_name == 'cifar':
            trainset = datasets.CIFAR10(
                root=path + '/datasets/CIFAR10', train=True, download=True, transform=train_transform)
            testset = datasets.CIFAR10(
                root=path + '/datasets/CIFAR10', train=False, download=True, transform=test_transform)

            valset = trainset
            # valset, val_x, val_y = get_valset(trainset, val_ratio)

            self.data_shape = (3, 32, 32)
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
        trainset, valset, testset = self.trainset, self.valset, self.testset

        # https://blog.csdn.net/SweetWind1996/article/details/105328385
        trainsampler = torch.utils.data.sampler.RandomSampler(trainset, replacement=False)
        valsampler = torch.utils.data.sampler.RandomSampler(valset, replacement=False)
        testsampler = torch.utils.data.sampler.RandomSampler(testset, replacement=False)

        # 所以这里才有明明batch大小是batch_size，但是下面loader里拿出来的是2*batch_size(batch)是因为对每一个都采样了两次

        multiplier = args_contains(self.args, 'multiplier', 1)
        batch_size = args_contains(self.args, 'batch_size', 200)
        val_batch_size = args_contains(self.args, 'val_batch_size', 50)
        test_batch_size = args_contains(self.args, 'test_batch_size', 2048)
        train_batch_sampler = MultiplyBatchSampler(
            multiplier, trainsampler, batch_size, drop_last=True)

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

        return train_loader, val_loader, test_loader
