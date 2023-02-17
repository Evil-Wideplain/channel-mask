import time
from utils.Util import init_parameters, save_record
from s_learning import S
from ss_learning import *

if __name__ == '__main__':
    # TODO 早期的参数配置 单独为 UCI USC MOTION三个数据集的
    # args = init_parameters('--version noise --cases random --batch_size 1024 --epochs 200 --GPU 1 --seed 888 '
    #                        '--framework simclr --backbone TPN --criterion NTXent '
    #                        '--aug1 resample,mask_sense --aug2 resample,mask_sense '
    #                        '--dataset Motion --data_dir .. --temperature 0.1 --weight_decay 0.0001 --p 96 --phid 96'
    #                        ' --lr 0.01 --ratio 0.01 --opt adam --lr_scheduler const --momentum 0.9 --multiplier 1 '
    #                        '--lr_cls 0.01 --fine_epochs 200 --val_batch_size 64 --test_batch_size 1024 '
    #                        '--fine_opt adam --mask_num 1 --mask_type drop'.split())

    args = cl_args
    args.save = True

    # TODO 1: random mask sensor channels
    # baseline(args, num=1)
    # CL_All_mask_random(args, num=1, mask_num=1.0)
    # CL_All_mask_random(args, num=1, mask_num=2.0)
    # CL_All_mask_random(args, num=1, mask_num=3.0)

    # TODO 2: analysis ucihar datasets with six channels
    # CL_uci_with_sixChannels(args, num=1)

    # TODO 3: random mask a proportion of sensor channels
    # CL_All_mask_rate(args, num=1, mask_num=1/6)
    # CL_All_mask_rate(args, num=1, mask_num=1/3)
    # CL_All_mask_rate(args, num=1, mask_num=1/2)
    # CL_All_mask_rate(args, num=1, mask_num=2/3)

    # TODO 4: analysis the effect of a certain single sensor axis
    # CL_certain_single_sensor_mask(args, num=1)
    # CL_certain_acc_sensor_device_mask(args, num=1)
    # CL_certain_gyr_sensor_device_mask(args, num=1)

    # TODO 5: test with augmentation
    CL_test_with_aug(args, num=1)

    # TODO 6: compare with Masked Reconstruction
    # CL_All_mask_features(args, num=1, mask_num=1/6)
    # CL_All_mask_features(args, num=1, mask_num=1/3)

