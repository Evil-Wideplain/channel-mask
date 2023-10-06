import time
from utils.Util import init_parameters, save_record
from Work import ContrastiveLearning as CL, SupervisedLearning as SL
from utils.Functional import Table


aug_mask_name = 'mask_sense_list'
# aug_mask_name = 'mask_sense_rate'
aug_list = [
    'na',
    aug_mask_name,
    'resample',
    'resample,' + aug_mask_name,
    aug_mask_name + ',resample',
    'noise',
    'noise,' + aug_mask_name,
    'resample,noise',
    'perm_jit',
    'perm,' + aug_mask_name]

sl_args = init_parameters('--version sl --cases random --epochs 120 --GPU 1 --seed 10 '
                           '--batch_size 256  --val_batch_size 256 --test_batch_size 256 '
                           '--framework simclr --backbone DCL --criterion NTXent '
                           '--aug1 na --aug2 na --dataset ucihar --ratio 0.5 '
                           '--data_dir .. --temperature 0.1 --weight_decay 1.e-6 --p 128 --phid 128 '
                           '--lr 0.003 --opt adam --lr_scheduler cosine '
                           '--lr_cls 0.003 --fine_epochs 120 --fine_opt adam '
                           '--momentum 0.9 --multiplier 1'.split())


def S(args=sl_args):
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
            # 每一个配置参数训练n次
            for _ in range(1):
                for aug in aug_list:
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

                    sl = SL(args)
                    train_loader, val_loader, test_loader = sl.train_loaders, sl.val_loader, sl.test_loader
                    acc = sl()
                    save_record("sl {} {} acc:{}".format(args.aug1, args.aug2, acc), './xieqi/record.txt')

