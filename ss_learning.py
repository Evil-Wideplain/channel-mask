import time
from utils.Util import init_parameters, save_record
from Work import ContrastiveLearning as CL, SupervisedLearning as SL
from utils.Functional import Table
import yaml

cl_args = init_parameters('--version mask_rate --cases random --epochs 120 --GPU 1 --seed 10 --save True '
                          '--batch_size 256  --val_batch_size 256 --test_batch_size 256 '
                          '--framework simclr --backbone FCN --criterion NTXent '
                          '--aug1 na --aug2 na --dataset ucihar --ratio 0.2,0.2 '
                          '--data_dir .. --temperature 0.1 --weight_decay 1.e-6 --p 128 --phid 128 '
                          '--lr 0.001 --opt adam --lr_scheduler cosine '
                          '--lr_cls 0.03 --fine_epochs 120 --fine_opt adam '
                          '--momentum 0.9 --multiplier 1 '
                          # channel mask
                          '--mask_num 1.0 --mask_type drop --mask_list 0 '
                          '--workers 2 --pin_memory True'.split())

# aug_mask_name = 'mask_sense_list'
# aug_mask_name = 'mask_sense_rate'
# aug_list = [
#     'na',
#     aug_mask_name,
#     'resample',
#     'resample,' + aug_mask_name,
#     aug_mask_name + ',resample',
#     'noise',
#     'noise,' + aug_mask_name,
#     'resample,noise',
#     'perm_jit',
#     'perm,' + aug_mask_name]

# , 'p_shift', 'p_shift,mask_sense'

sensor_list = ['acc_x', 'acc_y', 'acc_z', 'gry_x', 'gry_y', 'gry_z']
mask_lists_gyr = {
    'ucihar': '3,4,5',
    'deviceMotion': '3,4,5',
    'USCHAR': '3,4,5',
    'hhar': '3,4,5',
}

mask_lists_acc = {
    'ucihar': '0,1,2',
    'deviceMotion': '9,10,11',
    'USCHAR': '0,1,2',
    'hhar': '0,1,2',
}

mask_lists_single_given = {
    'ucihar': ['0', '1', '2', '3', '4', '5'],
    'deviceMotion': ['9', '10', '11', '3', '4', '5'],
    'USCHAR': ['0', '1', '2', '3', '4', '5'],
    'hhar': ['0', '1', '2', '3', '4', '5'],
}

_continue = []


def CL_Base(args, backbones, datasets, num, augs, augs_nums=3):
    save_record('---------------------------------------------------------------------------- start {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')

    for dataset in datasets:
        for backbone in backbones:
            args.backbone = backbone
            if "{}_{}".format(backbone, dataset) in _continue:
                continue
            for itr in range(num):
                save_record("{}[{}/{}]".format(dataset,
                            itr, num), './xieqi/record.txt')
                args.dataset = dataset
                table = Table(title="{}_{}".format(backbone, dataset), index=augs,
                              # columns=aug_list,
                              columns=['na/aug', 'aug/na',
                                       'aug/aug', 'average'],
                              sheet_name='table1')
                for aug in augs:
                    if aug == 'na':
                        continue
                    for s in range(augs_nums):
                        aug1 = 'na'
                        aug2 = 'na'
                        if s == 0:
                            aug1 = aug
                        elif s == 1:
                            aug2 = aug
                        elif s == 2:
                            aug1 = aug
                            aug2 = aug
                        else:
                            s = 3
                        args.model_path = ''
                        args.aug1 = aug1
                        args.aug2 = aug2

                        cl1 = CL(args)
                        path = cl1()
                        acc = cl1(path=[path])
                        save_record("aug1:{},aug2:{}  acc:{}".format(
                            aug1, aug2, acc), './xieqi/record.txt')
                        table.add(aug, s, acc)

                table.save_rect('./xieqi')
    save_record('---------------------------------------------------------------------------- end {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')


def baseline(args=cl_args, num=1):
    # mask_name = 'mask_sense'
    _aug = [
        'na', 'resample', 'noise', 'perm', 'scale',
        'resample,noise', 'perm_jit', 'jit_scal'
    ]
    args.mask_num = 1.0
    #
    CL_Base(args, backbones=['FCN'],
            datasets=['ucihar', 'shar', 'deviceMotion',
                      'pamap2', 'WISDM', 'USCHAR', 'DSADS', 'hhar'],
            num=num, augs=_aug)


def CL_All_mask_random(args=cl_args, num=1, mask_num=1.0):
    mask_name = 'mask_sense'
    _aug = [
        'na',
        mask_name,
        'resample,' + mask_name,
        # mask_name + ',resample',
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name
    ]
    args.mask_num = mask_num
    # , 'TPN'
    CL_Base(args, backbones=['FCN'],
            datasets=['ucihar', 'shar', 'deviceMotion',
                      'pamap2', 'WISDM', 'USCHAR', 'DSADS', 'hhar'],
            num=num, augs=_aug)


def CL_All_mask_rate(args=cl_args, num=1, mask_num=1/6):
    _continue = []
    mask_name = 'mask_sense_rate'
    _aug = [
        'na',
        mask_name,
        'resample,' + mask_name,
        # mask_name + ',resample',
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    args.mask_num = mask_num
    CL_Base(args, backbones=['FCN'], datasets=[
            'pamap2', 'DSADS', 'deviceMotion'], num=num, augs=_aug)


def CL_certain_single_sensor_mask(args=cl_args, num=1):
    global _continue
    _continue = ['FCN_ucihar']
    mask_name = 'mask_sense_list'
    augs = [
        'na',
        mask_name,
        'resample,' + mask_name,
        # mask_name + ',resample',
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    save_record('---------------------------------------------------------------------------- start {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')
    args.data_dir = '..'
    for backbone in ['FCN']:
        save_record(backbone, './xieqi/record.txt')
        args.backbone = backbone
        # , 'hhar'
        for dataset in ['ucihar', 'USCHAR', 'hhar', 'deviceMotion']:
            if dataset == 'ucihar':
                args.cases = 'random_six_channels'
            else:
                args.cases = 'random'
            if "{}_{}".format(backbone, dataset) in _continue:
                continue
            # 每一个配置参数训练n次
            for itr in range(num):
                save_record("{}[{}/{}]".format(dataset,
                            itr, num), './xieqi/record.txt')
                args.dataset = dataset
                for idx in range(6):
                    # if idx != 5:
                    #     continue
                    args.mask_list = mask_lists_single_given[dataset][idx]
                    save_record(sensor_list[idx], './xieqi/record.txt')
                    # print(args.mask_list)
                    table = Table(title="{}_{}".format(backbone, dataset), index=augs,
                                  # columns=aug_list,
                                  columns=['na/aug', 'aug/na',
                                           'aug/aug', 'average'],
                                  sheet_name='table1')
                    for aug in augs:
                        if aug == 'na':
                            continue
                        for s in range(3):
                            aug1 = 'na'
                            aug2 = 'na'
                            if s == 0:
                                aug2 = aug
                            elif s == 1:
                                aug1 = aug
                            elif s == 2:
                                aug1 = aug
                                aug2 = aug
                            else:
                                s = 3
                            args.model_path = ''
                            args.aug1 = aug1
                            args.aug2 = aug2

                            cl1 = CL(args)
                            path = cl1()
                            acc = cl1(path=[path])
                            save_record("aug1:{},aug2:{}  acc:{}".format(
                                aug1, aug2, acc), './xieqi/record.txt')
                            table.add(aug, s, acc)

                    table.save_rect('./xieqi')
    save_record('---------------------------------------------------------------------------- end {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')


def CL_certain_acc_sensor_device_mask(args=cl_args, num=1):
    global _continue
    _continue = []
    mask_name = 'mask_sense_list'
    augs = [
        mask_name,
        'resample,' + mask_name,
        # mask_name + ',resample',
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    save_record('---------------------------------------------------------------------------- start {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')
    args.data_dir = '..'
    for backbone in ['FCN']:
        save_record(backbone, './xieqi/record.txt')
        args.backbone = backbone
        for dataset in ['ucihar', 'USCHAR', 'deviceMotion', 'hhar']:
            if dataset == 'ucihar':
                args.cases = 'random_six_channels'
            else:
                args.cases = 'random'
            if "{}_{}".format(backbone, dataset) in _continue:
                continue
            # 每一个配置参数训练n次
            for itr in range(num):
                save_record("{}[{}/{}]".format(dataset,
                            itr, num), './xieqi/record.txt')
                args.dataset = dataset
                # TODO 加速度传感器
                args.mask_list = mask_lists_acc[dataset]
                save_record(args.mask_list, './xieqi/record.txt')
                # print(args.mask_list)
                table = Table(title="{}_{}".format(backbone, dataset), index=augs,
                              # columns=aug_list,
                              columns=['na/aug', 'aug/na',
                                       'aug/aug', 'average'],
                              sheet_name='table1')
                for aug in augs:
                    if aug == 'na':
                        continue
                    for s in range(3):
                        aug1 = 'na'
                        aug2 = 'na'
                        if s == 0:
                            aug2 = aug
                        elif s == 1:
                            aug1 = aug
                        elif s == 2:
                            aug1 = aug
                            aug2 = aug
                        else:
                            s = 3
                        args.model_path = ''
                        args.aug1 = aug1
                        args.aug2 = aug2

                        cl1 = CL(args)
                        path = cl1()
                        acc = cl1(path=[path])
                        save_record("aug1:{},aug2:{}  acc:{}".format(
                            aug1, aug2, acc), './xieqi/record.txt')
                        table.add(aug, s, acc)

                    table.save_rect('./xieqi')
    save_record('---------------------------------------------------------------------------- end {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')


def CL_certain_gyr_sensor_device_mask(args=cl_args, num=1):
    global _continue
    _continue = []
    mask_name = 'mask_sense_list'
    augs = [
        'na',
        mask_name,
        'resample,' + mask_name,
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    save_record('---------------------------------------------------------------------------- start {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')
    args.data_dir = '..'
    for backbone in ['FCN']:
        save_record(backbone, './xieqi/record.txt')
        args.backbone = backbone
        for dataset in ['ucihar', 'USCHAR', 'hhar', 'deviceMotion']:
            if dataset == 'ucihar':
                args.cases = 'random_six_channels'
            else:
                args.cases = 'random'
            if "{}_{}".format(backbone, dataset) in _continue:
                continue
            # 每一个配置参数训练n次
            for itr in range(num):
                save_record("{}[{}/{}]".format(dataset,
                            itr, num), './xieqi/record.txt')
                args.dataset = dataset
                # TODO 陀螺仪传感器
                args.mask_list = mask_lists_gyr[dataset]
                save_record(args.mask_list, './xieqi/record.txt')
                table = Table(title="{}_{}".format(backbone, dataset), index=augs,
                              # columns=aug_list,
                              columns=['na/aug', 'aug/na',
                                       'aug/aug', 'average'],
                              sheet_name='table1')
                for aug in augs:
                    if aug == 'na':
                        continue
                    for s in range(3):
                        aug1 = 'na'
                        aug2 = 'na'
                        if s == 0:
                            aug2 = aug
                        elif s == 1:
                            aug1 = aug
                        elif s == 2:
                            aug1 = aug
                            aug2 = aug
                        else:
                            s = 3
                        args.model_path = ''
                        args.aug1 = aug1
                        args.aug2 = aug2

                        cl1 = CL(args)
                        path = cl1()
                        acc = cl1(path=[path])
                        save_record("aug1:{},aug2:{}  acc:{}".format(
                            aug1, aug2, acc), './xieqi/record.txt')
                        table.add(aug, s, acc)

                    table.save_rect('./xieqi')
    save_record('---------------------------------------------------------------------------- end {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')


def CL_test_with_aug(args, num=1):
    save_record('---------------------------------------------------------------------------- start {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')
    mask_name = 'mask_sense'
    augs = [
        mask_name,
        'resample,' + mask_name,
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    args.mask_num = 1.0
    args.test_aug = True
    # args.data_dir = '..'
    for backbone in ['FCN']:  # 'DCL', 'FCN', 'TPN'
        save_record(backbone, './xieqi/record.txt')
        args.backbone = backbone
        for dataset in ['ucihar', 'shar', 'deviceMotion', 'pamap2', 'WISDM', 'USCHAR', 'DSADS', 'hhar']:
            if "{}_{}".format(backbone, dataset) in _continue:
                continue

            if dataset == 'ucihar':
                args.cases = 'random_six_channels'
            else:
                args.cases = 'random'
            # 每一个配置参数训练n次
            for itr in range(num):
                save_record("{}[{}/{}]".format(dataset,
                            itr, num), './xieqi/record.txt')
                args.dataset = dataset
                table = Table(title="{}_{}".format(backbone, dataset), index=augs,
                              # columns=aug_list,
                              columns=['na/aug', 'aug/na',
                                       'aug/aug', 'average'],
                              sheet_name='table1')
                for aug in augs:
                    if aug == 'na':
                        continue
                    for s in range(3):
                        aug1 = 'na'
                        aug2 = 'na'
                        if s == 0:
                            aug2 = aug
                        elif s == 1:
                            aug1 = aug
                        elif s == 2:
                            aug1 = aug
                            aug2 = aug
                        else:
                            s = 3
                        args.model_path = ''
                        args.aug1 = aug1
                        args.aug2 = aug2

                        cl1 = CL(args)
                        path = cl1()
                        acc = cl1(path=[path])
                        save_record("aug1:{},aug2:{}  acc:{}".format(
                            aug1, aug2, acc), './xieqi/record.txt')
                        table.add(aug, s, acc)

                table.save_rect('./xieqi')
    save_record('---------------------------------------------------------------------------- end {}'
                .format(time.strftime('%Y%m%d_%H%M%S', time.localtime())), './xieqi/record.txt')


def CL_uci_with_sixChannels(args=cl_args, num=1):
    mask_name = 'mask_sense'
    _aug = [
        'na',
        mask_name,
        'resample',
        'resample,' + mask_name,
        'noise',
        'noise,' + mask_name,
        'resample,noise',
        'perm_jit',
        'perm,' + mask_name,
        'scale', 'jit_scal',
        'scale,' + mask_name]
    args.mask_num = 1.0
    args.cases = 'random_six_channels'
    CL_Base(args, backbones=['FCN'],
            datasets=['ucihar'],
            num=num, augs=_aug)


def CL_All_mask_features(args=cl_args, num=1, mask_num=1/6):
    _continue = []
    mask_name = 'mask_features'
    _aug = [
        'na',
        mask_name,
        'resample,' + mask_name,
        'noise,' + mask_name,
        'perm,' + mask_name,
        'scale,' + mask_name]
    args.mask_num = mask_num
    CL_Base(args, backbones=['FCN'], datasets=['ucihar', 'shar', 'deviceMotion', 'pamap2', 'WISDM', 'USCHAR', 'DSADS'],
            num=num, augs=_aug)
