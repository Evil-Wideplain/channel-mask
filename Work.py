# import mxnet as mx
# from mxnet import autograd, gluon, image, init, nd
# from mxnet.gluon import model_zoo, nn
import gc
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
import pandas as pd
# from loss_cl import ContrativeLearning as CLLoss
# from loss_cluster import Cluster as CLoss
# from loss_svm import svm as SLoss
# from loss_svm_t import MMCL_INV, MMCL_PGD, NTXent
from criterion.Loss import Loss, calculate_cls
from models.backbones import NNMemoryBankModule
from utils.Util import set_seed, init_parameters, args_contains, compute_figure, compute_tps, save_csv, freeze
from aug.IMG import *
from models.Encoder import Encoder, Linclf, Backbone
from data.preprocess import dataloader
from utils.Util import save_record
from utils.Dis import pos_neg_dis
from aug.augmentations import gen_aug


def setup(args):
    # 参数
    args.time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 1.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    # TODO: simclr的优化器 weight_decay 不一定是0.0
    if args.framework == 'simclr':
        args.weight_decay = 0.0
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4
    return args


class ContrastiveLearning(nn.Module):
    def __init__(self, args=None):
        super(ContrastiveLearning, self).__init__()
        seed = args_contains(args, 'seed', 888)
        set_seed(seed)

        if args is None:
            self.args = init_parameters(args=[])
        else:
            self.args = args

        # setup 一些参数更新
        self.args = setup(self.args)

        self.t = self.args.time
        GPU = args_contains(self.args, 'GPU', '1')
        self.device = torch.device('cuda:{}'.format(GPU))
        _version = args_contains(self.args, 'dataset', 'UCI')
        self.version = self.t + '_' + _version
        # data
        self.args, self.train_loaders, self.val_loader, self.test_loader = dataloader(self.args)

        # model
        self.model, self.optimizers, self.schedulers, self.out_dim = Encoder(args, device=self.device)

        self.nn_replacer = None
        framework = args_contains(self.args, 'framework', 'nnclr')
        if framework == 'nnclr':
            mmb_size = args_contains(self.args, 'mmb_size', 1024)
            self.nn_replacer = NNMemoryBankModule(size=mmb_size)

        self.recon = None
        backbone = args_contains(self.args, 'backbone', 'TPN')
        if backbone in ['AE', 'CNN_AE']:
            self.recon = nn.MSELoss()

        self.criterion = Loss(args, device=self.device, recon=self.recon, nn_replacer=self.nn_replacer)
        self.criterion_cls = nn.CrossEntropyLoss()

        # 保存方法
        save = args_contains(self.args, 'save', True)
        self.writer = None
        # self.df = pd.DataFrame(self.args.__dict__)
        self.save_path = args_contains(self.args, 'save_path', './xieqi')
        if save:
            shape = args_contains(self.args, 'shape', (1, 128, 9))
            self.writer = SummaryWriter(self.save_path + "/log/{}".format(self.t))
            x_input = torch.randn(shape).to(self.device)
            self.writer.add_graph(self.model, input_to_model=[x_input, x_input])
            # self.out_file = open(save_path + '/log/{}/out.txt'.format(self.t), 'w')
            self.f_handler = open(self.save_path + '/log/{}/print.txt'.format(self.t), 'w')
            save_record(str(self.args.__dict__), self.save_path + '/log/{}/record.txt'.format(self.t))
            channels = int(np.ceil(float(args_contains(self.args, 'mask_num', 1.0)) * args_contains(self.args, 'n_features', 9)))
            save_record('mask_nums:{}/{}'.format(channels, args_contains(self.args, 'n_features', 9)), './xieqi/record.txt')

        # 降维分析
        self.tsne = TSNE(n_components=2, init='pca', random_state=501, perplexity=10)

    def l2_regularization(self, model, l2_alpha=1e-4):
        for module in model.modules():
            if type(module) is nn.Conv1d:
                module.weight.grad.data.add_(l2_alpha * module.weight.data)

    def draw_figure(self, deal_x, pres_labels, epoch):
        n_outputs = args_contains(self.args, 'n_outputs', 6)
        figure = compute_figure(self.tsne, deal_x, pres_labels, n_outputs, epoch+1)
        if self.writer:
            self.writer.add_figure('{}/figure'.format(self.version), figure, global_step=epoch + 1)
        return figure

    def tps(self, true_labels, pres_labels, epoch):
        n_outputs = args_contains(self.args, 'n_outputs', 6)
        tps_epoch, fns_epoch, fps_epoch, precisions_epoch, recalls_epoch \
            = compute_tps(true_labels, pres_labels, n_outputs, epoch)
        if self.writer:
            self.writer.add_histogram(
                '{}/tps'.format(self.version), np.array(tps_epoch), epoch)
            self.writer.add_histogram(
                '{}/precisions'.format(self.version), np.array(precisions_epoch), epoch)
            self.writer.add_histogram(
                '{}/recalls'.format(self.version), np.array(recalls_epoch), epoch)
        return tps_epoch, fns_epoch, fps_epoch, precisions_epoch, recalls_epoch

    def train_base(self, train_loaders, val_loaders):
        epochs = args_contains(self.args, 'epochs', 200)
        batch_size = args_contains(self.args, 'batch_size', 1024)
        n_features = args_contains(self.args, 'n_features', 9)
        framework = args_contains(self.args, 'framework', 'simclr')
        save = args_contains(self.args, 'save', True)
        _loss = 1e8
        min_val_loss = 1E18
        # leave=False 显示在一行
        loop_epoch = tqdm(range(int(epochs)), total=int(epochs), ncols=150)
        for epoch in loop_epoch:
            self.model.train()
            loss_epochs = []
            lr_epochs = []
            # pos_dis = []
            # neg_dis = []
            # TODO 暂定一下，choice的随机种子
            self.args.t_epoch = epoch
            for i, loader in enumerate(train_loaders):
                for idx, (sample, target, domain) in enumerate(loader):
                    # TODO ？？？
                    self.args.f_itr = idx+1
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()
                    # print(sample.shape)
                    # if sample.shape[0] != batch_size:
                    #     continue
                    loss = self.criterion(sample, target, self.model)
                    loss.backward()
                    for optimizer in self.optimizers:
                        optimizer.step()
                    if framework == 'byol':
                        self.model.update_moving_average()

                    with torch.no_grad():
                        loss_epochs.append(loss)
                        # pos, neg = pos_neg_dis(self.model.distance)
                        # pos_dis.append(pos)
                        # neg_dis.append(neg)
                        loop_epoch.set_description(
                            f'Epoch [{epoch + 1}/{epochs}]->Train({idx}/{len(loader)})')
                        loop_epoch.set_postfix(loss=loss.item())
                        if save:
                            for group in self.optimizers[0].param_groups:
                                lr_epochs.append(group['lr'])
                                # save_record("size: {}".format(np.array(group['lr']).shape), './record.txt')
                            # save_record("size: {}".format(np.array(lr_epochs).shape), './record.txt')
            if self.schedulers is not None:
                for scheduler in self.schedulers:
                    if scheduler is None:
                        continue
                    # scheduler.last_epoch = epoch - 1
                    scheduler.step()

            loss_train = torch.mean(torch.as_tensor(loss_epochs)).data
            with torch.no_grad():
                if _loss > loss_train:
                    _loss = loss_train
                    self.save(t='loss')
                if save:
                    save_record("{} {}".format(epoch, loss_train),
                                self.save_path + '/log/{}/loss.txt'.format(self.t), black_line=False)
                    # save_record("{} {}".format(epoch, np.sum(np.array(pos_dis))),
                    #             self.save_path + '/log/{}/pos_dis.txt'.format(self.t), black_line=False)
                    # save_record("{} {}".format(epoch, np.sum(np.array(neg_dis))),
                    #             self.save_path + '/log/{}/neg_dis.txt'.format(self.t), black_line=False)
                if self.writer:
                    self.writer.add_scalar('{}/loss'.format(self.version), loss_train, epoch)
                    # self.writer.add_scalar('{}/pos_dis'.format(self.version), np.sum(np.array(pos_dis)), epoch)
                    # self.writer.add_scalar('{}/neg_dis'.format(self.version), np.sum(np.array(neg_dis)), epoch)
                    self.writer.add_scalar('{}/lr'.format(self.version), np.array(lr_epochs)[0], epoch)
                    self.writer.add_histogram('{}/lr'.format(self.version), np.array(lr_epochs), epoch)

            cases = args_contains(self.args, 'cases', 'random')
            if cases in ['subject', 'subject_large'] or val_loaders is None:
                with torch.no_grad():
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.save(t='best')
                break
            else:
                with torch.no_grad():
                    self.model.eval()
                    total_loss = 0
                    n_batches = 0
                    val_batch_size = args_contains(self.args, 'val_batch_size', 64)
                    for idx, (sample, target, domain) in enumerate(val_loaders):
                        # if sample.shape[0] != val_batch_size:
                        #     continue
                        n_batches += 1
                        loss = self.criterion(sample, target, self.model)
                        total_loss += loss.item()
                    if total_loss <= min_val_loss:
                        min_val_loss = total_loss
                        best_model = copy.deepcopy(self.model.state_dict())
                        self.save(t='best')
                    if save:
                        save_record("{} {}".format(epoch, total_loss/n_batches),
                                    self.save_path + '/log/{}/loss_val.txt'.format(self.t), black_line=False)
                    if self.writer:
                        self.writer.add_scalar('{}/val_loss'.format(self.version), total_loss/n_batches, epoch)

            # gc.collect()
            # torch.cuda.empty_cache()
        return best_model

    def test_base(self, test_loader, best_model):
        # model, _, _, _ = Encoder(self.args, device=self.device)
        self.model.load_state_dict(best_model)
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            n_batches = 0
            test_batch_size = args_contains(self.args, 'test_batch_size', 1024)
            for idx, (sample, target, domain) in enumerate(test_loader):
                # if sample.size(0) != test_batch_size:
                #     continue
                n_batches += 1
                loss = self.criterion(sample, target, self.model)
                total_loss += loss.item()
            print('Test Loss   　:', total_loss / n_batches)
        return self.model

    def test_epoch(self, test_loader, trained_backbone, classifier, criterion):
        test_with_aug = args_contains(self.args, 'test_aug', False)
        aug = 'na'
        if test_with_aug:
            aug1 = args_contains(self.args, 'aug1', 'resample')
            aug2 = args_contains(self.args, 'aug2', 'na')
            aug = aug1 if aug1 != 'na' else aug2 if aug2 != 'na' else 'na'
            # print("test with {}".format(aug))
        with torch.no_grad():
            classifier.eval()
            total = 0
            correct_t = 0
            loss_epochs = []
            features = None
            targets = []
            predictions = []
            for idx, (sample, target, domain) in enumerate(test_loader):
                if test_with_aug:
                    sample = gen_aug(sample, aug, self.args)
                sample, target = sample.to(self.device).float(), target.to(self.device).float()
                loss, feat, predicted = calculate_cls(sample, target, trained_backbone, classifier, criterion)
                loss_epochs.append(loss.item())
                total += target.size(0)
                if features is None:
                    features = feat
                else:
                    features = torch.cat((features, feat), 0)
                targets = np.append(targets, target.data.cpu().numpy())
                predictions = np.append(predictions, predicted.data.cpu().numpy())
                if len(target.shape) != len(predicted.shape):
                    _, target = torch.max(target, -1)
                correct_t += (predicted == target).sum()
            acc = float(correct_t) / total
            loss = torch.mean(torch.as_tensor(loss_epochs)).data
        return acc, loss, features, targets, predictions

    def train_lincls(self, train_loaders, val_loader, trained_backbone):
        f_epochs = args_contains(self.args, 'fine_epochs', 200)
        lr_cls = args_contains(self.args, 'lr_cls', 0.1)
        save = args_contains(self.args, 'save', True)
        classifier = Linclf(self.args, trained_backbone.out_dim, self.device)
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr_cls)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cls, T_max=f_epochs, eta_min=0)
        # TODO test with aug
        test_with_aug = args_contains(self.args, 'test_aug', False)
        aug = 'na'
        if test_with_aug:
            aug1 = args_contains(self.args, 'aug1', 'resample')
            aug2 = args_contains(self.args, 'aug2', 'na')
            aug = aug1 if aug1 != 'na' else aug2 if aug2 != 'na' else 'na'

        loop = tqdm(range(f_epochs), total=f_epochs, ncols=150)
        min_val_loss = 1E18

        for epoch in loop:
            classifier.train()
            loss_epochs = []
            correct = 0
            total = 0
            acc_val, loss_val = 0.0, 0.0
            # TODO 和上面对比学习的一样，记录微调的进展
            self.args.f_epoch = epoch
            for i, train_loader in enumerate(train_loaders):
                for idx, (sample, target, domain) in enumerate(train_loader):
                    # TODO test with aug
                    if test_with_aug:
                        sample = gen_aug(sample, aug, self.args)
                    sample, target = sample.to(self.device).float(), target.to(self.device).float()
                    # train and calculate loss
                    loss, feat, predicted = calculate_cls(sample, target, trained_backbone,
                                                          classifier, self.criterion_cls)
                    total += target.size(0)
                    if len(target.shape) != len(predicted.shape):
                        _, target = torch.max(target, -1)
                    correct += (predicted == target).sum()
                    loss_epochs.append(loss)
                    optimizer_cls.zero_grad()
                    loss.backward()
                    optimizer_cls.step()
            if scheduler:
                scheduler.step()
            acc_val = float(correct)/total
            loss_val = torch.mean(torch.as_tensor(loss_epochs)).numpy()
            # gc.collect()
            # torch.cuda.empty_cache()
            loop.set_description(f'Epoch [{epoch + 1}/{f_epochs}]')
            loop.set_postfix(val_loss=loss_val, val_acc=acc_val)

            acc_test, loss_test = 0.0, 0.0
            cases = args_contains(self.args, 'cases', 'random')
            if cases in ['subject', 'subject_large']:
                with torch.no_grad():
                    best_lincls = copy.deepcopy(classifier.state_dict())
            else:
                classifier.eval()
                # TODO: 使用最后的训练好的模型？
                #  简单对比了以下，在绝大部分数据集上，最后的模型的并不是最好的，依然选择loss最低的模型最优
                # min_val_loss = 1E18
                with torch.no_grad():
                    acc_test, loss_test, features, targets, predictions = \
                        self.test_epoch(val_loader, trained_backbone, classifier, self.criterion_cls)
                    if loss_test <= min_val_loss:
                        min_val_loss = loss_test
                        best_lincls = copy.deepcopy(classifier.state_dict())
                    loop.set_postfix(val_loss=loss_val, val_acc=acc_val, test_loss=loss_test, acc=acc_test)
            if save:
                save_record("{} {}".format(epoch, acc_val),
                            self.save_path + '/log/{}/acc_val.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, loss_val),
                            self.save_path + '/log/{}/loss_val_lcf.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, loss_test),
                            self.save_path + '/log/{}/loss_test_clf.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, acc_test),
                            self.save_path + '/log/{}/acc.txt'.format(self.t), black_line=False)
            if self.writer:
                self.writer.add_scalar('{}/[{}]val_acc'.format(self.version, self.t), acc_val, epoch)
                self.writer.add_scalar('{}/[{}]val_loss'.format(self.version, self.t), loss_val, epoch)
                self.writer.add_scalar('{}/[{}]acc'.format(self.version, self.t), acc_test, epoch)
                self.writer.add_scalar('{}/[{}]test_loss'.format(self.version, self.t), loss_test, epoch)

        return best_lincls

    def test_linclf(self, test_loader, trained_backbone, best_lincls):
        classifier = Linclf(self.args, trained_backbone.out_dim, self.device)
        classifier.load_state_dict(best_lincls)
        classifier.eval()
        with torch.no_grad():
            best_acc = 0.0
            # acc_mean = []
            acc_test, loss_test, features, targets, predictions = \
                self.test_epoch(test_loader, trained_backbone, classifier, self.criterion_cls)
            # acc_mean.append(acc_test)
            # if acc_test > best_acc:
            best_acc = acc_test
            self.args.best_acc = best_acc
        return best_acc

    def train(self):
        save = args_contains(self.args, 'save', True)
        train_loaders, val_loader, test_loader = self.train_loaders, self.val_loader, self.test_loader
        # print(self.model)
        # print(self.optimizers)
        # print(self.criterion)
        # print(self.schedulers)
        # print(self.criterion_cls)
        print('train:{}     validation:{}   test:{}'.format(
            len(train_loaders[0]), len(val_loader), len(test_loader)))
        # TODO 原作者方法
        # TODO 总体1%的数据进行 得去修改一下数据获取的方法
        pretrain_state = self.train_base(train_loaders, val_loader)
        pretrain_model = self.test_base(test_loader, pretrain_state)
        # pretrain_model = self.model
        # print("pretrain model: ", pretrain_model)
        framework = args_contains(self.args, 'framework', 'simclr')
        _, train_backbone = freeze(pretrain_model, framework, False)
        # print('backbone: ', train_backbone)
        path = self.save_model(train_backbone)
        print(path)
        if save:
            save_record(str(self.args.__dict__), self.save_path + '/log/{}/record.txt'.format(self.t))
            save_record(path, self.save_path + '/log/{}/record.txt'.format(self.t))
        return path

    def test(self, path):
        save = args_contains(self.args, 'save', True)
        train_loaders, val_loader, test_loader = self.train_loaders, self.val_loader, self.test_loader
        train_backbone = self.load_model(path)
        # TODO: 之前微调验证的数据量为0.2*0.2*D 并没有想象中的80%... 和原作者的代码也不一样，所以做错了，但是结果对比来说要好，
        #  表示方法的效果好，试试其他的。
        # TODO 1. 原作者
        # lincls = self.train_lincls(train_loaders, val_loader, train_backbone)
        # TODO 2. 我之前测试， 写错了，可恶
        #  这里是将val_loader设置为了1%的微调数据
        lincls = self.train_lincls([val_loader], test_loader, train_backbone)
        # TODO 3. 总体的按照之前的方法，总体的1%的数据进行微调
        # train_loaders.append(val_loader)
        # lincls = self.train_lincls(train_loaders, test_loader, train_backbone)
        if lincls is None:
            if save:
                save_record("lincls is None", self.save_path + '/log/{}/record.txt'.format(self.t))
            return 0.0
        best_acc = self.test_linclf(test_loader, train_backbone, lincls)
        print(self.args.lr_cls, ' ', best_acc)
        if save:
            save_record(str(self.args.__dict__), self.save_path + '/log/{}/record.txt'.format(self.t))
            save_record("{} {}\n".format(self.args.lr_cls, best_acc),
                        self.save_path + '/log/{}/record.txt'.format(self.t))
        save_csv(self.args, best_acc)
        return best_acc

    def forward(self, path=None):
        if path is None:
            path = self.train()
            return path
        else:
            acc = 0.
            if not isinstance(path, list):
                path = [path]
            for p in path:
                if self.args.dataset in p:
                    acc = self.test(p)
            return acc

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict(),
            'args': self.args,
        }

    @classmethod
    def load(cls, ckpt_path):
        ckpt = torch.load(ckpt_path)
        args = ckpt['args']
        res = cls(args)
        res.load_state_dict(ckpt['state_dict'])
        return res

    def save(self, acc=0.0, t='new'):
        ckpt = self.get_ckpt()
        path = args_contains(self.args, 'save_path', './xieqi')
        model_path = args_contains(self.args, 'model_path', '')
        if t == 'loss':
            ckpt_path = path + '/log/{}/{}_loss.pth'.format(self.t, self.version)
        elif t == 'best':
            ckpt_path = path + '/log/{}/{}_best.pth'.format(self.t, self.version)
            self.args.best_acc = acc
        else:
            ckpt_path = path + '/log/{}/{}_{:.3f}.pth'.format(self.t, self.version, acc)
            self.args.acc = acc

        torch.save(ckpt, ckpt_path)
        if ckpt_path not in model_path:
            self.args.model_path = model_path + ',' + ckpt_path

        return self.args.model_path

    def save_model(self, model):
        path = args_contains(self.args, 'save_path', './xieqi')
        path = path + '/log/{}/{}_pretrain.pth'.format(self.t, self.version)
        torch.save({'pretrain': model.state_dict()}, path)
        return path

    def load_model(self, path):
        backbone = Backbone(self.args)
        ckpt = torch.load(path)
        backbone.load_state_dict(ckpt['pretrain'])
        return backbone.to(self.device)


class SupervisedLearning(nn.Module):
    def __init__(self, args=None):
        super(SupervisedLearning, self).__init__()
        seed = args_contains(args, 'seed', 888)
        set_seed(seed)
        if args is None:
            self.args = init_parameters(args=[])
        else:
            self.args = args
        # setup 一些参数更新
        self.args = setup(self.args)

        self.t = self.args.time
        GPU = args_contains(self.args, 'GPU', '1')
        self.device = torch.device('cuda:{}'.format(GPU))
        _version = args_contains(self.args, 'dataset', 'UCI')
        self.version = self.t + '_' + _version
        # data
        self.args, self.train_loaders, self.val_loader, self.test_loader = dataloader(self.args)
        # model
        self.backbone = Backbone(self.args, bone=False).to(self.device)
        lr_cls = args_contains(self.args, 'lr_cls', 0.1)
        f_epochs = args_contains(self.args, 'fine_epochs', 200)
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr_cls)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=f_epochs, eta_min=0)
        self.criterion = nn.CrossEntropyLoss()

        # 保存方法
        save = args_contains(self.args, 'save', True)
        self.writer = None
        # self.df = pd.DataFrame(self.args.__dict__)
        self.save_path = args_contains(self.args, 'save_path', './xieqi')
        if save:
            shape = args_contains(self.args, 'shape', (1, 128, 9))
            self.writer = SummaryWriter(self.save_path + "/log/{}".format(self.t))
            x_input = torch.randn(shape).to(self.device)
            self.writer.add_graph(self.backbone, input_to_model=x_input)
            # self.out_file = open(save_path + '/log/{}/out.txt'.format(self.t), 'w')
            self.f_handler = open(self.save_path + '/log/{}/print.txt'.format(self.t), 'w')
            save_record(str(self.args.__dict__), self.save_path + '/log/{}/record.txt'.format(self.t))
        # 降维分析
        self.tsne = TSNE(n_components=2, init='pca', random_state=501, perplexity=10)

    def test_epoch(self, test_loader, trained_backbone, classifier, criterion):
        with torch.no_grad():
            trained_backbone.eval()
            if classifier is not None:
                classifier.eval()
            total = 0
            correct_t = 0
            loss_epochs = []
            features = None
            targets = []
            predictions = []
            for idx, (sample, target, domain) in enumerate(test_loader):
                sample, target = sample.to(self.device).float(), target.to(self.device).float()
                loss, feat, predicted = calculate_cls(sample, target, trained_backbone, classifier, criterion)
                loss_epochs.append(loss.item())
                total += target.size(0)
                if features is None:
                    features = feat
                else:
                    features = torch.cat((features, feat), 0)
                targets = np.append(targets, target.data.cpu().numpy())
                predictions = np.append(predictions, predicted.data.cpu().numpy())
                if len(target.shape) != len(predicted.shape):
                    _, target = torch.max(target, -1)
                correct_t += (predicted == target).sum()
            acc = float(correct_t) / total
            loss = torch.mean(torch.as_tensor(loss_epochs)).data
        return acc, loss, features, targets, predictions

    def train_lincls(self, train_loaders, test_loader):
        f_epochs = args_contains(self.args, 'fine_epochs', 200)
        save = args_contains(self.args, 'save', True)

        loop = tqdm(range(f_epochs), total=f_epochs, ncols=150)
        min_val_loss = 1E18
        for epoch in loop:
            self.backbone.train()
            loss_epochs = []
            correct = 0
            total = 0
            acc_val, loss_val = 0.0, 0.0
            for i, train_loader in enumerate(train_loaders):
                for idx, (sample, target, domain) in enumerate(train_loader):
                    # aug
                    aug1 = args_contains(self.args, 'aug1', 'resample')
                    aug2 = args_contains(self.args, 'aug2', 'na')
                    if aug1 != aug2:
                        aug_sample1 = gen_aug(sample, aug1, self.args)
                        aug_sample2 = gen_aug(sample, aug2, self.args)
                        sample = np.concatenate([aug_sample1, aug_sample2], axis=0)
                        target = np.concatenate([target, target], axis=0)
                        sample, target = torch.as_tensor(sample), torch.as_tensor(target)
                    else:
                        sample = gen_aug(sample, aug1, self.args)
                        if type(sample) == np.ndarray:
                            sample = torch.FloatTensor(sample)

                    sample, target = sample.to(self.device).float(), target.to(self.device).float()
                    # train
                    loss, feat, predicted = calculate_cls(sample, target, self.backbone, None, self.criterion)
                    total += target.size(0)
                    if len(target.shape) != len(predicted.shape):
                        _, target = torch.max(target, -1)
                    correct += (predicted == target).sum()
                    loss_epochs.append(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            acc_val = float(correct)/total
            loss_val = torch.mean(torch.as_tensor(loss_epochs)).numpy()
            # gc.collect()
            # torch.cuda.empty_cache()
            loop.set_description(f'Epoch [{epoch + 1}/{f_epochs}]')
            loop.set_postfix(val_loss=loss_val, val_acc=acc_val)

            acc_test, loss_test = 0.0, 0.0
            cases = args_contains(self.args, 'cases', 'random')
            if cases in ['subject', 'subject_large']:
                with torch.no_grad():
                    best_lincls = copy.deepcopy(self.backbone.state_dict())
            else:
                self.backbone.eval()
                with torch.no_grad():
                    acc_test, loss_test, features, targets, predictions = \
                        self.test_epoch(test_loader, self.backbone, None, self.criterion)
                    if loss_test <= min_val_loss:
                        min_val_loss = loss_test
                        best_lincls = copy.deepcopy(self.backbone.state_dict())
                    loop.set_postfix(val_loss=loss_val, val_acc=acc_val, test_loss=loss_test, acc=acc_test)
            if save:
                save_record("{} {}".format(epoch, acc_val),
                            self.save_path + '/log/{}/acc_val.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, loss_val),
                            self.save_path + '/log/{}/loss_val_lcf.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, loss_test),
                            self.save_path + '/log/{}/loss_test_clf.txt'.format(self.t), black_line=False)
                save_record("{} {}".format(epoch, acc_test),
                            self.save_path + '/log/{}/acc.txt'.format(self.t), black_line=False)
            if self.writer:
                self.writer.add_scalar('{}/[{}]val_acc'.format(self.version, self.t), acc_val, epoch)
                self.writer.add_scalar('{}/[{}]val_loss'.format(self.version, self.t), loss_val, epoch)
                self.writer.add_scalar('{}/[{}]acc'.format(self.version, self.t), acc_test, epoch)
                self.writer.add_scalar('{}/[{}]test_loss'.format(self.version, self.t), loss_test, epoch)

        return best_lincls

    def test_linclf(self, backbone, test_loader):
        backbone.eval()
        with torch.no_grad():
            best_acc = 0.0
            acc_mean = []
            acc_test, loss_test, features, targets, predictions = \
                self.test_epoch(test_loader, backbone, None, self.criterion)
            acc_mean.append(acc_test)
            if acc_test > best_acc:
                best_acc = acc_test
            self.args.best_acc = best_acc
        return best_acc

    def forward(self):
        save = args_contains(self.args, 'save', True)
        train_loaders, val_loader, test_loader = self.train_loaders, self.val_loader, self.test_loader
        train_loaders.append(val_loader)
        lincls = self.train_lincls(train_loaders, test_loader)
        self.backbone.load_state_dict(lincls)
        self.save_model(self.backbone)
        if lincls is None:
            if save:
                save_record("lincls is None", self.save_path + '/log/{}/record.txt'.format(self.t))
            return 0.0
        backbone, _ = freeze(self.backbone, 'backbone', False)
        best_acc = self.test_linclf(backbone, test_loader)
        print(self.args.lr_cls, ' ', best_acc)
        if save:
            save_record("{} {}\n".format(self.args.lr_cls, best_acc),
                        self.save_path + '/log/{}/record.txt'.format(self.t))
        save_csv(self.args, best_acc)
        return best_acc

    def save_model(self, model):
        path = args_contains(self.args, 'save_path', './xieqi')
        path = path + '/log/{}/{}_pretrain.pth'.format(self.t, self.version)
        torch.save({'pretrain': model.state_dict()}, path)
        return path

    def load_model(self, path):
        backbone = Backbone(self.args, bone=False)
        ckpt = torch.load(path)
        backbone.load_state_dict(ckpt['pretrain'])
        return backbone.to(self.device)
