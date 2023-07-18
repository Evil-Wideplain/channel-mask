from torch.nn import LSTM

from utils.Util import args_contains
from models.frameworks import *
from models.backbones import *
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
        backbone = FCN(n_channels=n_feature, n_timesteps=n_timesteps,
                       n_classes=n_class, backbone=bone)
    else:
        NotImplementedError

    return backbone


def Framework(args, backbone, device):
    model = None
    n_feature = args_contains(args, 'n_features', 9)
    n_timesteps = args_contains(args, 'n_timesteps', 128)
    n_class = args_contains(args, 'n_class', 6)
    framework_name = args_contains(args, 'framework', 'simclr')
    p = args_contains(args, 'p', 128)
    if framework_name == 'simclr':
        model = SimCLR(backbone=backbone, dim=p)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif framework_name == 'sl':
        model = SupervisedLearning(backbone, n_class)
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
    decay = args_contains(args, 'decay_list', ' ').split()
    return get_optimizers(model=model, args=args,
                          opt=opt, lr=lr, lr_scheduler=lr_scheduler,
                          momentum=momentum, warmup=warmup,
                          weight_decay=weight_decay, decay_list=decay)


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
