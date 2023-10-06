import torch
from torch import nn
from torch.nn import functional as F
from utils.Util import args_contains
from utils.SVM import get_mask, get_kmask, compute_kernel, pgd_with_nesterov, pgd_simple_short


class SvmLoss(nn.Module):
    def __init__(self, args, device=torch.device('cuda:0')):
        super(SvmLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, xis, xjs):
        zis = F.normalize(xis, p=2, dim=1)
        zjs = F.normalize(xjs, p=2, dim=1)
        bs = zis.shape[0]

        kernel_type = args_contains(self.args, 'kernel_type', 'rbf')
        sigma = args_contains(self.args, 'sigma', 0.1)
        K = compute_kernel(zis, torch.cat([zis, zjs], dim=0), kernel_type=kernel_type, gamma=sigma)

        with torch.no_grad():
            block = torch.zeros(bs, 2 * bs).bool().to(self.device)
            block[:bs, :bs] = True
            KK = torch.masked_select(K.detach(), block).reshape(bs, bs)

            no_diag = (1 - torch.eye(bs)).bool().to(self.device)
            KK_d0 = KK * no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1, bs, 1)
            KXY = KXY + KXY.transpose(2, 1)

            reg = args_contains(self.args, 'reg', 0.1)
            oneone = (torch.ones(bs, bs) + torch.eye(bs) * reg).to(self.device)
            Delta = (oneone + KK).unsqueeze(0) + KXY

            DD_KMASK = get_kmask(bs, device=self.device)
            DD = torch.masked_select(Delta, DD_KMASK).reshape(bs, bs - 1, bs - 1)

            C = args_contains(self.args, 'C', 1.0)
            if C == -1:
                alpha_y = torch.relu(torch.randn(bs, bs - 1, 1, device=DD.device))
            else:
                alpha_y = torch.relu(torch.randn(
                    bs, bs - 1, 1, device=DD.device)).clamp(min=0, max=C)

            solver_type = args_contains(self.args, 'solver_type', 'nesterov')
            eta = args_contains(self.args, 'eta', 1e-3)
            num_iter = args_contains(self.args, 'num_iter', 1000)
            use_norm = args_contains(self.args, 'use_norm', True)
            stop_condition = args_contains(self.args, 'stop_condition', 1e-2)
            one_bs = torch.ones(bs, bs - 1, 1).to(self.device)
            if solver_type == 'nesterov':
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_with_nesterov(
                    eta, num_iter, DD, 2 * one_bs, alpha_y.clone(), C, use_norm=use_norm,
                    stop_condition=stop_condition)
            elif solver_type == 'vanilla':
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_simple_short(
                    eta, num_iter, DD, 2 * one_bs, alpha_y.clone(), C, use_norm=use_norm,
                    stop_condition=stop_condition)

            alpha_y = alpha_y.squeeze(2)
            if C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=C).detach()
            alpha_x = alpha_y.sum(1)

        block12 = torch.zeros(bs, 2 * bs).bool().to(self.device)
        block12[:bs, bs:] = True
        Ks = torch.masked_select(K, block12).reshape(bs, bs)

        anchor_count = args_contains(self.args, 'anchor_count', 2)

        mask, logits_mask = get_mask(bs, anchor_count)
        eye = torch.eye(anchor_count * bs).to(self.device)
        pos_mask = mask[:bs, bs:].bool()
        neg_mask = (mask * logits_mask + 1) % 2
        neg_mask = neg_mask - eye
        neg_mask = neg_mask[:bs, bs:].bool()
        Kn = torch.masked_select(Ks.T, neg_mask).reshape(bs, bs - 1).T

        pos_loss = (alpha_x * (Ks * pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T * Kn).sum() / bs

        # 琢磨
        loss = torch.exp(neg_loss - pos_loss)
        # loss = neg_loss - pos_loss

        sparsity = (alpha_y == C).sum() / ((alpha_y > 0).sum() + 1e-10)
        num_zero = (alpha_y == 0).sum() / alpha_y.numel()
        # (Ks * pos_mask).sum(1).mean(), Kn.mean(), sparsity, num_zero, 0.0
        return loss

