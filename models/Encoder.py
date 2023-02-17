from torch.nn import LSTM

from utils.Util import args_contains
from models.frameworks import *
from models.backbones import *
from models.backbones2 import *
from optim.Optimizer import get_optimizers


def Backbone(args, bone=True):
    backbone = None
    backbone_name = args_contains(args, 'backbone', 'TPN')
    n_feature = args_contains(args, 'n_features', 9)
    n_class = args_contains(args, 'n_class', 6)
    n_timesteps = args_contains(args, 'n_timesteps', 128)
    dataset = args_contains(args, 'dataset', 'UCI')
    # set up backbone network
    if backbone_name == 'FCN':
        backbone = FCN(n_channels=n_feature, n_timesteps=n_timesteps, n_classes=n_class, backbone=bone)
    elif backbone_name == 'DCL':
        backbone = DeepConvLSTM(n_channels=n_feature, n_classes=n_class, conv_kernels=64, kernel_size=5,
                                LSTM_units=128, backbone=bone)
    elif backbone_name == 'LSTM':
        backbone = LSTM(n_channels=n_feature, n_classes=n_class, LSTM_units=128, backbone=bone)
    elif backbone_name == 'AE':
        backbone = AE(n_channels=n_feature, len_sw=n_timesteps, n_classes=n_class, outdim=128, backbone=bone)
    elif backbone_name == 'CNN_AE':
        backbone = CNN_AE(n_channels=n_feature, n_classes=n_class, out_channels=128, backbone=bone)
    elif backbone_name == 'Transformer':
        backbone = Transformer(n_channels=n_feature, len_sw=n_timesteps, n_classes=n_class, dim=128, depth=4,
                               heads=4, mlp_dim=64, dropout=0.1, backbone=bone)
    elif backbone_name == 'TPN':
        if dataset == 'opportunity':
            kernel_list = [16, 8, 4]
            channels_size = None
        else:
            channels_size = None
            kernel_list = None
        backbone = TPN(in_channels=n_feature, n_classes=n_class, channels_size=channels_size, kernel_list=kernel_list,
                       backbone=bone)
    elif backbone_name == 'ResNet_Z':
        num_blocks = [2, 2, 2, 2]
        backbone = ResNet_Z(block=BasicBlock, num_blocks=num_blocks, n_classes=n_class)
    else:
        NotImplementedError

    return backbone


def Framework(args, backbone, device):
    model = None
    n_feature = args_contains(args, 'n_features', 9)
    n_timesteps = args_contains(args, 'n_timesteps', 128)
    framework_name = args_contains(args, 'framework', 'simclr')
    phid = args_contains(args, 'phid', 128)
    p = args_contains(args, 'p', 128)
    if framework_name in ['byol', 'simsiam']:
        model = BYOL(device, backbone, window_size=n_timesteps, n_channels=n_feature, projection_size=p,
                     projection_hidden_size=phid, moving_average=EMA)
        # optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
        #                               lr,
        #                               weight_decay=weight_decay)
        # optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
        #                               lr * lr_mul,
        #                               weight_decay=weight_decay)
    elif framework_name == 'simclr':
        model = SimCLR(backbone=backbone, dim=p)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif framework_name == 'nnclr':
        model = NNCLR(backbone=backbone, dim=p, pred_dim=phid)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizers = [optimizer]
    elif framework_name == 'tstcc':
        temp_unit = args_contains(args, 'temp_unit', 'tsfm')
        model = TSTCC(backbone=backbone, DEVICE=device, temp_unit=temp_unit, tc_hidden=100)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        # optimizers = [optimizer]
    else:
        NotImplementedError

    return model


def get_opts(args, model):
    opt = args_contains(args, 'opt', 'adam')
    lr = args_contains(args, 'lr', 0.001)
    lr_mul = args_contains(args, 'lr_mul', 10.0)
    weight_decay = args_contains(args, 'weight_decay', 1.e-06)
    lr_scheduler = args_contains(args, 'lr_scheduler', 'const')
    momentum = args_contains(args, 'momentum', 0.9)
    warmup = args_contains(args, 'warmup', 0.01)
    return get_optimizers(model=model, args=args,
                          opt=opt, LR=lr, lr_scheduler=lr_scheduler,
                          momentum=momentum, warmup=warmup,
                          weight_decay=weight_decay)


def Linclf(args, bb_dim, device):
    """
    :param args: init parameters
    :param device: GPU device
    :param bb_dim: output dimension of the backbone network
    :return a linear classifier
    """
    n_class = args_contains(args, 'n_class', 6)
    classifier = Classifier(bb_dim=bb_dim, n_classes=n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(device)
    return classifier


def Encoder(args, device=torch.device("cuda:0")):
    backbone = Backbone(args)
    model = Framework(args, backbone, device)
    classifier = None

    framework_name = args_contains(args, 'framework', 'simclr')
    if framework_name in ['byol', 'simsiam']:
        optimizer1, scheduler1 = get_opts(args, model.online_encoder)
        optimizer2, scheduler2 = get_opts(args, model.online_predictor)
        optimizers = [optimizer1, optimizer2]
        schedulers = [scheduler1, scheduler2]
    else:
        optimizer, scheduler = get_opts(args, model)
        optimizers = [optimizer]
        schedulers = [scheduler]

    model = model.to(device)

    # if cls:
    #     bb_dim = backbone.out_dim
    #     classifier = Linclf(args, bb_dim, device)
    # return model, bb_dim, optimizers, schedulers
    # else:
    return model, optimizers, schedulers, backbone.out_dim
