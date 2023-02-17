import torch
import numpy as np

def get_kmask(bs, device):
    KMASK = torch.ones(bs, bs, bs).bool().to(device)
    for t in range(bs):
        KMASK[t, t, :] = False
        KMASK[t, :, t] = False
    return KMASK.detach()


def get_mask(batch_size, anchor_count, device=torch.device('cuda:0')):
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    mask = mask.repeat(anchor_count, anchor_count)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    return mask, logits_mask


def compute_kernel(X, Y, gamma=0.1, kernel_type='rbf', device=torch.device('cuda:0')):
    kernel = None
    if kernel_type == 'linear':
        kernel = torch.mm(X, Y.T)
    elif kernel_type == 'rbf':
        if gamma == 'auto':
            gamma = 1 / X.shape[-1]
        gamma = 1. / float(gamma)
        # gamma 增大，轮廓就越贴合样本点。
        # gamma主要定义了单个样本对整个分类超平面的影响。
        # 当q比较小时，单个样本对整个分类超平面的影响比较小，不容易被选择为支持向量
        # 反之，当q比较大时，单个样本对整个分类超平面的影响比较大，更容易被选择为支持向量，或者说整个模型的支持向量也会多
        distances = -gamma * torch.cdist(X, Y).pow(2)
        # distances = -gamma*(2-2.*torch.mm(X, Y.T))
        kernel = torch.exp(distances)
    elif kernel_type == 'poly':
        kernel = torch.pow(torch.mm(X, Y.T) + 0.5, 3.)
    elif kernel_type == 'tanh':
        kernel = torch.tanh(gamma * torch.mm(X, Y.T))
    elif kernel_type == 'min':
        # kernel = torch.minimum(torch.relu(X), torch.relu(Y))
        kernel = torch.min(torch.relu(X).unsqueeze(
            1), torch.relu(Y).unsqueeze(1).transpose(1, 0)).sum(2)
    else:
        kernel = None
    return kernel


def pgd_simple_short(eta, num_iter, Q, p, alpha_y, C, use_norm=False, stop_condition=0.01):
    if use_norm:
        eta_to_use = 1.0/torch.norm(Q, dim=(1, 2), p=2, keepdim=True)
    else:
        eta_to_use = eta
    theta1 = torch.eye(Q.shape[1], device=Q.device).unsqueeze(
        0).repeat(Q.shape[0], 1, 1) - eta_to_use*Q
    theta2 = eta_to_use*p
    rel_change_init = -1.0
    for iter_no in range(num_iter):

        x_new = torch.bmm(theta1, alpha_y) + theta2
        if C == -1:
            alpha_y_new = torch.relu(x_new).detach()
        else:
            alpha_y_new = torch.relu(x_new).clamp(min=0, max=C)

        abs_rel_change = ((alpha_y_new-alpha_y)/(alpha_y + 1E-7)).abs().mean()

        if iter_no == 0:
            rel_change_init = abs_rel_change
        if abs_rel_change < stop_condition:
            alpha_y = alpha_y_new
            break
        alpha_y = alpha_y_new

    return alpha_y, iter_no, abs_rel_change, rel_change_init


def pgd_with_nesterov(eta, num_iter, Q, p, alpha_y, C, use_norm=False, stop_condition=0.01):
    import time
    start_timer = time.time()
    if use_norm:
        eta_to_use = 1.0/torch.norm(Q, dim=(1, 2), p=2, keepdim=True)
    else:
        eta_to_use = eta

    theta1 = torch.eye(Q.shape[1], device=Q.device).unsqueeze(
        0).repeat(Q.shape[0], 1, 1) - eta_to_use*Q
    theta2 = eta_to_use*p
    alpha0 = np.random.uniform(low=1E-8)
    y = alpha_y
    rel_change_init = -1.0
    for iter_no in range(num_iter):

        x_new = torch.bmm(theta1, y) + theta2

        if C == -1:
            alpha_y_new = torch.relu(x_new).detach()
        else:
            alpha_y_new = torch.relu(x_new).clamp(min=0, max=C)

        alpha0_new = 0.5*(np.sqrt(alpha0**4 + 4*alpha0**2) - alpha0**2)
        beta_k = (alpha0*(1-alpha0))/(alpha0**2 + alpha0_new)
        y = alpha_y_new + beta_k*(alpha_y_new - alpha_y)
        alpha0 = alpha0_new

        abs_rel_change = ((alpha_y_new-alpha_y)/(alpha_y + 1E-7)).abs().mean()
        if iter_no == 0:
            rel_change_init = abs_rel_change
        if abs_rel_change < stop_condition:
            alpha_y = alpha_y_new
            break
        alpha_y = alpha_y_new

    return alpha_y, iter_no, abs_rel_change, rel_change_init
