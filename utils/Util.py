import os
import logging
import sys

from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 模型结构保持不变，输入大小保持不变，能使得cudNN能够计算许多不同的卷积计算方法，然后使用最快的方法
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass

    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_torch(seed=1029):
    import random as pyton_random
    pyton_random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_data_parameter(parser):
    # data
    parser.add_argument('--dataset', default='UCI', type=str)
    parser.add_argument('--data_dir', default='..', type=str)
    parser.add_argument('--ratio', default='0.2,0.2', type=str)
    parser.add_argument('--feature_dimensions', default=128, type=int)
    parser.add_argument('--train_iter', default=0, type=int)
    parser.add_argument('--val_iters', default=0, type=int)
    parser.add_argument('--test_iters', default=0, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--multiplier', default=1, type=int,
                        help="data process need, data augmentation if need in dataloader")
    # num_workers 最好的是GPU数量的2倍 用的单卡跑的代码，那么就是2
    parser.add_argument('--workers', default=0, type=int)
    # pin_memory
    parser.add_argument('--pin_memory', default=False, type=bool)
    parser.add_argument('--shape', default=(1, 128, 9), type=tuple)
    parser.add_argument('--n_timesteps', default=128, type=int)
    parser.add_argument('--n_features', default=9, type=int)
    parser.add_argument('--n_class', default=6, type=int)
    # 如果要将时序数据转图像
    parser.add_argument('--toImage', default=False, type=bool)
    parser.add_argument('--n_width', default=64, type=int)
    parser.add_argument('--n_height', default=64, type=int)

    # augmentation
    parser.add_argument('--aug1', type=str, default='resample,mask_sense',
                        help='the type of augmentation transformation')
    parser.add_argument('--aug2', type=str, default='resample,mask_sense',
                        help='the type of augmentation transformation')
    # sense feature mask
    parser.add_argument('--mask_num', default=1.0, type=float)
    parser.add_argument('--mask_type', default='drop',
                        type=str, choices=['drop', 'noise'])
    parser.add_argument('--mask_list', default="0,", type=str)
    # test with aug
    parser.add_argument('--test_aug', default=False, type=bool)
    # insert_noise
    parser.add_argument('--noise_alpha', default=0.2, type=float)

    return parser


def add_optimizer_parameter(parser):
    # optimizer
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--fine_opt', default='adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_cls', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1.e-06, type=float)
    parser.add_argument('--lr_scheduler', default='const', type=str)
    parser.add_argument('--decay_list', default=' ', type=str)
    return parser


def add_framework_parameter(parser):
    # encoder
    parser.add_argument('--framework', default='simclr', type=str,
                        choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'sl'])
    parser.add_argument('--backbone', default='FCN', type=str,
                        choices=['FCN'])
    parser.add_argument('--criterion', type=str, default='NTXent',
                        choices=['NTXent'])
    parser.add_argument('--p', type=int, default=128,
                        help='byol: projector size, simsiam: projector output size, simclr: projector output size')
    parser.add_argument('--phid', type=int, default=128,
                        help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')
    return parser


def add_other_parameters(parser):
    parser.add_argument('--distance', default='euclid', type=str,
                        choices=['euclid', 'manhattan', 'cosine_similarity'])
    parser.add_argument('--acc', default=0.0, type=float)
    parser.add_argument('--best_acc', default=0.0, type=float)
    return parser


def init_parameters(args=None, parent_parser=None):
    import argparse
    if parent_parser is None:
        parser = argparse.ArgumentParser(
            description="self-supervised cluster for HAR")
    else:
        parser = parent_parser
    # parser.add_argument('--type', default='train', type=str, choices=['train', 'fine', 'linear', 'supervised'])
    parser.add_argument('--time', default='now', type=str)
    parser.add_argument('--version', default='cluster', type=str)
    parser.add_argument('--GPU', default=1, type=int)
    parser.add_argument('--device', default='Phones',
                        choices=['Phones', 'Watch'], type=str)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--save_path', default='./xieqi', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--target_domain', type=str, default='0')
    parser.add_argument('--cases', type=str, default='random',
                        choices=['random', 'random_fine', 'subject', 'cross_device', 'joint_device'])
    parser.add_argument('--seed', default=888, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--t_epoch', default=0, type=int, help="用来描述的当前的epoch")
    # cls
    parser.add_argument('--fine_epochs', default=200, type=int)
    parser.add_argument('--f_epoch', default=0,
                        type=int, help="用来描述的当前微调的epoch")
    parser.add_argument('--f_itr', default=0, type=int,
                        help="记录dataloader的第几个batch,怎么会有人这样做呢？")
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=2048, type=int)

    parser = add_framework_parameter(parser)
    parser = add_data_parameter(parser)
    parser = add_optimizer_parameter(parser)
    parser = add_other_parameters(parser)

    if args is None:
        args = ""
    return parser.parse_args(args=args)


def save_csv(args, acc=0.0):
    path = args_contains(args, 'save_path', './xieqi')
    args.acc = acc
    args.shape = [str(args.shape)]
    # 将参数保存到 config.csv表格中
    if os.path.exists(path + '/log/result.csv'):
        df = pd.read_csv(path + '/log/result.csv',
                         encoding="unicode_escape", index_col=0)
        # df1 = pd.DataFrame(args.__dict__).iloc[0]
        df1 = pd.DataFrame(args.__dict__)
        df = pd.concat([df, df1])
    else:
        df = pd.DataFrame(args.__dict__)
    df.to_csv(path + '/log/result.csv')
    return path + '/log/result.csv'


def save_record(content, path, black_line=True):
    # print(type(content[0]))
    if not isinstance(content, list):
        if not isinstance(content, str):
            content = str(content)
        content = [content]

    txt_object = open(path, "a+")
    if black_line:
        txt_object.write("\n")
    txt_object.write('\n'.join(content) + '\n')
    txt_object.close()


def args_has(args, name):
    return hasattr(args, name)


def args_get(args, name):
    return getattr(args, name)


def args_contains(args, name, default):
    if hasattr(args, name):
        return getattr(args, name)
    else:
        return default


def handler_sysout(handler=None):
    if handler is None:
        return handler
    import sys
    oldstdout = sys.stdout
    sys.stdout = handler
    return oldstdout


def compute_figure(tsne, deal_x, pres_labels, n_outputs, epoch):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ts = [[] for _ in range(n_outputs)]
    colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#5D4037',
              '#e50000', '#95d0fc', '#029386', '#f97306', '#FFC107', '#FFC107']
    # if len(deal_x.shape) == 1:
    # print(deal_x.shape)
    # deal_x.reshape(-1, 1)

    tsne_d = tsne.fit_transform(deal_x)
    for (i, data) in enumerate(zip(tsne_d, pres_labels)):
        x, y = data
        ts[np.int(y)].append(x)

    for (i, data) in enumerate(ts):
        data = np.array(data)
        if data.shape[0] != 0:
            plt.scatter(data[:100, 0], data[:100, 1],
                        marker='.', alpha=0.8, c=colors[i], label=str(i))

    plt.grid()
    plt.legend(loc='upper left')
    plt.title('epoch' + str(epoch))
    # plt.show()
    return figure


def compute_tps(true_labels, pres_labels, n_outputs, epoch):
    tps_epoch = []
    fns_epoch = []
    fps_epoch = []
    precisions_epoch = []
    recalls_epoch = []
    for c in range(n_outputs):
        # 总共有多少样本
        a = len(true_labels)
        # 预测结果中样本标签为`l`的有
        p_l = np.sum(pres_labels == c)
        # 样本标签为`l`的 正类有
        t_l = np.sum(true_labels == c)
        # 样本中标签为l的且被预测标签也为l的

        tp = 0
        for _i in range(a):
            if pres_labels[_i] == true_labels[_i] == c:
                tp = tp + 1

        # print(np.sum(pres_labels == true_labels == c))
        # 样本中标签为了的但是被预测标签不是l的
        fn = t_l - tp
        # 样本中标签不是l但是被预测为l
        fp = p_l - tp
        # 剩下的，样本标签不是l且也没有被预测为l的
        # 正例+反例=样本数
        # tp + fn = 真正例
        # fp + tn = 真反例
        # tp + fp = 预测正例
        # fn + tn = 预测反例
        tn = a - fp - t_l
        precision = np.float(np.float(tp) / (tp + fp + 1e-8))
        recall = np.float(np.float(tp) / (tp + fn + 1e-8))
        tps_epoch.append(tp)
        fns_epoch.append(fn)
        fps_epoch.append(fp)
        precisions_epoch.append(precision)
        recalls_epoch.append(recall)

        # if self.writer:
        #     self.writer.add_scalar(
        #         '{}/类别_{}_tp'.format(self.version, c), tp, epoch)
        #     self.writer.add_scalar(
        #         '{}/类别_{}_precision'.format(self.version, c), precision, epoch)
        #     self.writer.add_scalar(
        #         '{}/类别_{}_recall'.format(self.version, c), recall, epoch)

    return tps_epoch, fns_epoch, fps_epoch, precisions_epoch, recalls_epoch


def freeze(model, framework, tf=False):
    for name, param in model.named_parameters():
        param.requires_grad = tf

    if framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    else:
        trained_backbone = model

    return model, trained_backbone


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    # y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16, 10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        legend="full",
        alpha=0.5
    )

    sns_plot.get_figure().savefig(save_dir)


def mds(latent, y_ground_truth, save_dir):
    """
        Plot MDS embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    mds = MDS(n_components=2)
    mds_results = mds.fit_transform(latent)
    plt.figure(figsize=(16, 10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=mds_results[:, 0], y=mds_results[:, 1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full",
        alpha=0.5
    )

    sns_plot.get_figure().savefig(save_dir)
