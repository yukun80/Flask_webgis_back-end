import math
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction="none")(
        temp_inputs, temp_target
    )
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta**2) * tp + smooth) / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    计算 Lovasz 扩展的梯度
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(jaccard) == 1:
        p = 1
    else:
        p = jaccard[1:] - jaccard[:-1]
    return p

def lovasz_hinge(logits, labels, ignore=None):
    """
    Binary Lovasz hinge loss
    logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
    labels: [P] Tensor, binary ground truth labels (0 or 1)
    ignore: label to ignore
    """
    # Flatten logits and labels and filter out ignored elements
    logits, labels = flatten_binary_scores(logits, labels, ignore)

    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)

    # Ensure grad and errors_sorted are the same length
    if len(grad) != len(errors_sorted):
        grad = torch.cat([grad, grad[-1].repeat(len(errors_sorted) - len(grad))])

    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def Lovasz_Hinge_Loss(inputs, target, num_classes=2):
    """
    Lovasz Hinge Loss 函数
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # 对于二分类任务，我们只需要类别的一个维度
    # 这里我们假设正类的 logits 在第二维（索引为1）
    logits = temp_inputs[:, 1]

    lovasz_loss = lovasz_hinge(logits, temp_target)

    return lovasz_loss


def Combined_Loss(inputs, target, cls_weights, lovasz_weight=0.5, ce_weight=0.5, num_classes=2):
    """
    结合 Lovasz Hinge Loss 和 Cross Entropy Loss 的损失函数
    :param inputs: 模型的输出 logits
    :param target: 真实的标签
    :param cls_weights: 类别权重，用于 Cross Entropy Loss
    :param lovasz_weight: Lovasz Loss 的权重
    :param ce_weight: Cross Entropy Loss 的权重
    :param num_classes: 类别数量
    :return: 结合后的总损失
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 对于 Cross Entropy Loss
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    ce_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)

    # 对于 Lovasz Hinge Loss
    # 假设正类的 logits 在第二维（索引为1）
    logits = temp_inputs[:, 1]
    lovasz_loss = lovasz_hinge(logits, temp_target)

    # 结合两种损失
    combined_loss = ce_weight * ce_loss + lovasz_weight * lovasz_loss
    return combined_loss


def weights_init(net, init_type="kaiming", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s type" % init_type)
    net.apply(init_func)


def get_lr_scheduler(
    lr_decay_type,
    lr,
    min_lr,
    total_iters,
    warmup_iters_ratio=0.05,
    warmup_lr_ratio=0.1,
    no_aug_iter_ratio=0.05,
    step_num=10,
):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate**n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
