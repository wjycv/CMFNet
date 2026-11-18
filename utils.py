# -*- coding: utf-8 -*-


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


# 修改adjust_lr函数（假设在utils.py中）
# def adjust_lr(optimizer, init_lr, epoch, decay_rate, decay_epoch, warmup_epochs=0):
#     if warmup_epochs > 0 and epoch <= warmup_epochs:
#         # Warmup阶段：线性增加学习率
#         lr = init_lr * (epoch / warmup_epochs)
#     else:
#         # 正常衰减阶段：从warmup结束后开始计算衰减
#         if epoch <= warmup_epochs:
#             lr = init_lr
#         else:
#             step = (epoch - warmup_epochs - 1) // decay_epoch
#             lr = init_lr * (decay_rate ** step)
#     # 更新优化器的学习率
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr