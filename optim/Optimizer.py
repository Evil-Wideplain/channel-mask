import os
# import mxnet as mx
# from mxnet import autograd, gluon, image, init, nd
# from mxnet.gluon import model_zoo, nn
import torch
from torch import nn
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ConstantLR, ExponentialLR, \
    SequentialLR, CosineAnnealingLR, ChainedScheduler, ReduceLROnPlateau, CyclicLR, \
    CosineAnnealingWarmRestarts
# LinearLR

import numpy as np
from utils.Util import args_contains
from optim.lars import LARS
from optim.LRScheduler import LinearLR, LinearWarmupAndCosineAnneal


def get_optimizers(model, args,
                   opt='lars', lr=1.2,
                   lr_scheduler='warmup-anneal',
                   momentum=0.9, warmup=0.01, betas=(0.9, 0.999),
                   weight_decay=1.0e-06, decay_list=None,
                   iters=200  # (self.iters == 0 if len(self.train_loader) else self.iters)
                   ):
    # 判断是否需要使用权重衰减
    if decay_list is None:
        # ['bn', 'relu', 'dropout', 'bias', 'lars', 'fc', 'globalMaxPool']
        decay_list = []

    def exclude_from_wd_and_adaptation(name):
        for n in decay_list:
            if n in name:
                return True
        # if 'bn' in name or 'relu' in name or 'dropout' in name:
        #     return True
        # if 'bias' in name:
        #     return True
        # if opt == 'lars' and 'bias' in name:
        #     return True
        # if 'fc' in name:
        #     return True
        # if 'globalMaxPool' in name or 'flatten' in name:
        #     return True
        return False

    # 需要学习和调整的模型层
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    # param_name_groups = [
    #     {
    #         'params': [name for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
    #         'weight_decay': weight_decay,
    #         'layer_adaptation': True,
    #     },
    #     {
    #         'params': [name for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
    #         'weight_decay': 0.,
    #         'layer_adaptation': False,
    #     },
    # ]
    # print("optimizer param_groups: {}".format(param_name_groups))

    if opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
        )
    elif opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
        )
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum
        )
    elif opt == 'lars':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
        )
        lars_optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if lr_scheduler == 'warmup-anneal':
        scheduler = LinearWarmupAndCosineAnneal(
            optimizer,
            warmup,
            iters,
            last_epoch=-1,
        )
    elif lr_scheduler == 'linear':
        scheduler = LinearLR(optimizer, iters, last_epoch=-1)
    elif lr_scheduler == 'cosine':
        epoch = args_contains(args, 'epochs', 200)
        eta_min = args_contains(args, 'eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=eta_min)
    elif lr_scheduler == 'cosine-restart':
        epoch = args_contains(args, 'epochs', 200)
        eta_min = args_contains(args, 'eta_min', 0)
        T_0 = args_contains(args, 'T_0', 0)
        T_mult = args_contains(args, 'T_mult', 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    elif lr_scheduler == 'const':
        # TODO: 16 学习率调整策略 单独聚类方法不需要
        scheduler = None
    else:
        raise NotImplementedError

    if opt == 'lars':
        optimizer = lars_optimizer

    return optimizer, scheduler
