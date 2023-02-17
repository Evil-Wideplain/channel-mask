import time
from utils.Util import init_parameters, save_record
from Work import ContrastiveLearning as CL, SupervisedLearning as SL
from utils.Functional import Table

def S(args):
    # 每一个配置参数训练n次
    for _ in range(1):
        save_record('---------------------------------------------------------------------------- 训练数据0.5',
                    './xieqi/record.txt')
        args.save = True
        args.data_dir = '..'
        for backbone in ['DCL', 'FCN', 'TPN']:  # 'DCL', 'FCN', 'TPN'
            save_record(backbone, './xieqi/record.txt')
            args.backbone = backbone
            # for lr in [0.1, 0.01, 0.001, 0.0001]:
            # for lr in [0.001, 0.0001]:
            #     args.lr = lr
            # 'UCI', 'USC', 'Motion'
            # 'ucihar', 'shar', 'deviceMotion', 'opportunity', 'pamap2', 'WISDM'
            for dataset in ['USCHAR', 'hhar']:
                save_record(dataset, './xieqi/record.txt')
                args.dataset = dataset

                for aug in [
                    {
                        'aug1': 'na',
                        'aug2': 'na'
                    },
                    # {
                    #     'aug1': 'resample',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'mask_sense',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'noise',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'resample,noise',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'resample,mask_sense',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'perm_jit',
                    #     'aug2': 'na'
                    # },
                    {
                        'aug1': 'na',
                        'aug2': 'resample'
                    },
                    {
                        'aug1': 'na',
                        'aug2': 'mask_sense'
                    },
                    {
                        'aug1': 'na',
                        'aug2': 'noise'
                    },
                    {
                        'aug1': 'na',
                        'aug2': 'resample,noise'
                    },
                    {
                        'aug1': 'na',
                        'aug2': 'resample,mask_sense'
                    },
                    {
                        'aug1': 'na',
                        'aug2': 'perm_jit'
                    },
                    {
                        'aug1': 'resample',
                        'aug2': 'resample'
                    },
                    {
                        'aug1': 'mask_sense',
                        'aug2': 'mask_sense'
                    },
                    {
                        'aug1': 'noise',
                        'aug2': 'noise'
                    },
                    {
                        'aug1': 'resample,noise',
                        'aug2': 'resample,noise'
                    },
                    {
                        'aug1': 'resample,mask_sense',
                        'aug2': 'resample,mask_sense'
                    },
                    {
                        'aug1': 'perm_jit',
                        'aug2': 'perm_jit'
                    },
                    # {
                    #     'aug1': 'mask_sense,resample',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'na',
                    #     'aug2': 'mask_sense,resample'
                    # },
                    # {
                    #     'aug1': 'mask_sense,resample',
                    #     'aug2': 'mask_sense,resample'
                    # },
                    # {
                    #     'aug1': 'noise,mask_sense',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'na',
                    #     'aug2': 'noise,mask_sense'
                    # },
                    # {
                    #     'aug1': 'noise,mask_sense',
                    #     'aug2': 'noise,mask_sense'
                    # },
                    # {
                    #     'aug1': 'insert_noise',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'na',
                    #     'aug2': 'insert_noise'
                    # },
                    # {
                    #     'aug1': 'insert_noise',
                    #     'aug2': 'insert_noise'
                    # },
                    # {
                    #     'aug1': 'resample,insert_noise',
                    #     'aug2': 'na'
                    # },
                    # {
                    #     'aug1': 'na',
                    #     'aug2': 'resample,insert_noise'
                    # },
                    # {
                    #     'aug1': 'resample,insert_noise',
                    #     'aug2': 'resample,insert_noise'
                    # },
                ]:
                    args.model_path = ''
                    args.aug1 = aug['aug1']
                    args.aug2 = aug['aug2']

                    sl = SL(args)
                    train_loader, val_loader, test_loader = sl.train_loaders, sl.val_loader, sl.test_loader
                    acc = sl()
                    save_record("sl {} {} acc:{}".format(args.aug1, args.aug2, acc), './xieqi/record.txt')

