# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import sys

from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.CMFNet import pvt

from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt

# 判断GPU是否可用
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# 加载训练集路径
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
# 加载验证集路径
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path
# 保存log日志
if not os.path.exists(save_path):
    os.makedirs(save_path)
logging.basicConfig(filename=save_path + 'mobile_vit.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("mobile_vit_pairs")
# 创建模型
model = pvt()

# 加载backbone的与训练参数
num_parms = 0
if opt.load_rgb is not None:
    model.load_pre(opt.load_rgb, opt.load_depth)

    print('load rgb_model from ', opt.load_rgb)
    print('load depth_model from ', opt.load_depth)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
# 优化器
optimizer = torch.optim.Adam(params, opt.lr)

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=3e-4,
#     weight_decay=1e-3
# )
# 创建文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 加载训练集数据
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# 加载验证集数据
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
# 求训练集数据长度
total_step = len(train_loader)
# 写日志
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load_rgb:{};load_depth:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_rgb, opt.load_depth,
        save_path,
        opt.decay_epoch))
# 定义损失函数计算函数
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# 定义IOU损失
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def balance_loss(gating_weights):
    # 计算每个专家的平均使用率
    expert_usage = torch.mean(gating_weights, dim=(0, 2, 3))
    # 计算平衡损失
    loss = torch.var(expert_usage)
    return loss


# 边缘损失
class EdgeLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 从GT提取边缘
        edge_gt = F.max_pool2d(target, 3, stride=1, padding=1) - \
                  F.avg_pool2d(target, 3, stride=1, padding=1)
        edge_gt = (edge_gt > 0.1).float()

        # 从预测结果提取边缘
        edge_pred = pred.sigmoid()
        edge_pred = F.max_pool2d(edge_pred, 3, stride=1, padding=1) - \
                    F.avg_pool2d(edge_pred, 3, stride=1, padding=1)

        # 计算边缘区域损失
        intersection = (edge_pred * edge_gt).sum()
        union = edge_pred.sum() + edge_gt.sum() + self.epsilon
        return 1 - (2 * intersection / union)


# 层级感知损失
class HierarchyAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = iou_loss  # 使用原有iou_loss
        self.edge = EdgeLoss()

        # 各层损失权重配置
        self.layer_weights = {
            'x1': {'bce': 0.2, 'dice': 0.2, 'edge': 0.6},
            # 'x1': {'bce': 0.3, 'dice': 0.4, 'edge': 0.3},
            'x2': {'bce': 0.3, 'dice': 0.5, 'edge': 0.2},
            'x3': {'bce': 0.4, 'dice': 0.6, 'edge': 0.0},
            'x4': {'bce': 0.5, 'dice': 0.5, 'edge': 0.0}
            # 'x4': {'bce': 0.6, 'dice': 0.4, 'edge': 0.0}
        }

    def forward(self, preds, target):
        out1, out2, out3, out4 = preds

        # 计算各层损失
        loss_x1 = (
                self.layer_weights['x1']['bce'] * self.bce(out1, target) +
                self.layer_weights['x1']['dice'] * self.dice(out1, target) +
                self.layer_weights['x1']['edge'] * self.edge(out1, target)
        )

        loss_x2 = (
                self.layer_weights['x2']['bce'] * self.bce(out2, target) +
                self.layer_weights['x2']['dice'] * self.dice(out2, target) +
                self.layer_weights['x2']['edge'] * self.edge(out2, target)
        )

        loss_x3 = (
                self.layer_weights['x3']['bce'] * self.bce(out3, target) +
                self.layer_weights['x3']['dice'] * self.dice(out3, target)
        )

        loss_x4 = (
                self.layer_weights['x4']['bce'] * self.bce(out4, target) +
                self.layer_weights['x4']['dice'] * self.dice(out4, target)
        )

        # 总损失（层级权重可调）
        total_loss = 0.2 * loss_x1 + 0.1 * loss_x2 + 0.2 * loss_x3 + 0.5 * loss_x4
        return total_loss, (loss_x1, loss_x2, loss_x3, loss_x4)





# 开始训练
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    # 初始化层级感知损失
    criterion = HierarchyAwareLoss().cuda()

    for i, (images, gts, depth) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = images.cuda()
        gts = gts.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()

        # 前向传播
        out1, out2, out3, out4, balance_loss_total = model(images, depth, return_gating_weights=True)

        # 计算层级感知损失
        total_loss, (loss1, loss2, loss3, loss4) = criterion((out1, out2, out3, out4), gts)
        # 加入平衡损失
        total_loss += balance_loss_total * 0.1  # 平衡损失权重系数

        # 反向传播
        total_loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1
        # 最终损失和
        loss_all += total_loss.data
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss1:{:4f}, loss2:{:4f}, loss3:{:4f}, '
                  'loss4:{:4f}, balance_loss:{:4f}, loss:{:4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         optimizer.state_dict()['param_groups'][0]['lr'], loss1.data, loss2.data, loss3.data, loss4.data, balance_loss_total.data, total_loss.data))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss1:{:4f}, loss2:{:4f}, '
                'loss3:{:4f}, loss4:{:4f}, balance_loss:{:4f}, loss:{:4f} mem_use:{:.0f}MB'.
                format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                       loss1.data, loss2.data, loss3.data, loss4.data, balance_loss_total.data, total_loss.data, memory_used))
            writer.add_scalar('Loss', total_loss.data, global_step=step)
            grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('Ground_truth', grid_image, step)


            res4 = out4[0].clone()
            res4 = res4.sigmoid().data.cpu().numpy().squeeze()
            res4 = (res4 - res4.min()) / (res4.max() - res4.min() + 1e-8)
            writer.add_image('res4', torch.tensor(res4), step, dataformats='HW')

    loss_all /= epoch_step
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    # 每5轮保存一次模型参数
    if (epoch) % 20 == 0:
        torch.save(model.state_dict(), save_path + 'mobile_vit_epoch_{}.pth'.format(epoch))


# 验证
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_x1 = 0
        mae_x2 = 0
        mae_x3 = 0
        mae_x4 = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            x1, x2, x3, x4 = model(image, depth)


            res4 = F.upsample(x4, size=gt.shape, mode='bilinear', align_corners=False)
            res4 = res4.sigmoid().data.cpu().numpy().squeeze()
            res4 = (res4 - res4.min()) / (res4.max() - res4.min() + 1e-8)
            # 计算MAE
            mae_sum += np.sum(np.abs(res4 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        # mae_sum = (mae_x1 + mae_x2 + mae_x3 + mae_x4) / 4
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        # 进行MAE比较
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'mobile_vit_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))




if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
