import time
from utils.Util import init_parameters, save_record
from s_learning import S
from ss_learning import *

if __name__ == '__main__':
    args = cl_args
    args.save = True
    args.GPU = 0

    # TODO 训练按1/6比例遮盖
    # CL_All_mask_rate(args, num=1, mask_num=1/3)

    # 1: random mask sensor channels
    # baseline(args, num=1)
    # CL_All_mask_random(args, num=1, mask_num=1.0)
    # CL_All_mask_random(args, num=1, mask_num=2.0)
    # CL_All_mask_random(args, num=1, mask_num=3.0)

    # 2: analysis ucihar datasets with six channels
    # CL_uci_with_sixChannels(args, num=1)

    # 3: random mask a proportion of sensor channels
    # CL_All_mask_rate(args, num=1, mask_num=1/6)
    CL_All_mask_rate(args, mask_num=1/3)
    # CL_All_mask_rate(args, num=1, mask_num=1/2)
    # CL_All_mask_rate(args, num=1, mask_num=2/3)

    # 4: analysis the effect of a certain single sensor axis
    # CL_certain_single_sensor_mask(args, num=1)
    # CL_certain_acc_sensor_device_mask(args, num=1)
    # CL_certain_gyr_sensor_device_mask(args, num=1)

    # 5: test with augmentation
    # CL_test_with_aug(args, num=1)

    # 6: compare with Masked Reconstruction
    # CL_All_mask_features(args, num=1, mask_num=1/6)
    # CL_All_mask_features(args, num=1, mask_num=1/3)

    # 7: more
    # more(args)
    
