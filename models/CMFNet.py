# -*- coding: utf-8 -*-

import os
from functools import partial

import cv2
import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math
from mobilevit import mobile_vit_xx_small
from mobilevit import mobile_vit_small
from mobilevit import mobile_vit_x_small
from torchvision.ops import DeformConv2d


def upsample(x, y):
    return F.interpolate(x, y, mode='bilinear', align_corners=True)


class BConv(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, h, w, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, w))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(kernel_size=self.sz, stride=self.sz)
        self.gmpLayer = nn.MaxPool2d(kernel_size=self.sz, stride=self.sz)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg


class sa_layer(nn.Module):
    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out


class DynamicFusion(nn.Module):
    def __init__(self, channel, h, w):
        super().__init__()
        self.channel = channel
        self.h = h
        self.w = w

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)


        # 原始组件保持不变
        self.conv_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * 2, channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 9 * 2 * channel, 1),
            nn.Unflatten(1, (channel, 9 * 2))
        )
        self.convTo2 = nn.Conv2d(channel * 2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.coord_att = CoordAtt(channel, channel, h, w)
        self.glbamp = AP_MP()
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 2 + 1, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.base_conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

        # === 新增自适应注意力模块 ===
        self.fusion_att = nn.Sequential(
            nn.Conv2d(channel * 2, channel // 8, 3, padding=1),  # 输入双倍通道
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1),  # 输出单通道权重图
            nn.Sigmoid()
        )

    def _dynamic_conv(self, combined):
        # 保持原有动态卷积实现不变
        batch, _, h, w = combined.size()
        conv_weights = self.conv_gen(combined)
        weight_bank = conv_weights.view(batch * self.channel, 2, 3, 3)
        combined_reshaped = combined.view(1, batch * 2 * self.channel, h, w)
        dynamic_feat = F.conv2d(
            combined_reshaped,
            weight_bank,
            padding=1,
            groups=batch * self.channel
        )
        return dynamic_feat.view(batch, self.channel, h, w)

    def forward(self, r, d):
        # 动态卷积路径
        combined = torch.cat([r, d], dim=1)
        dynamic_feat = self._dynamic_conv(combined)

        # 原始Fusion处理路径
        H = torch.cat((r, d), dim=1)
        H_conv = self.sig(self.convTo2(H))
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        Ga = r * ga
        Gm = d * gm

        Gm_out = self.coord_att(Gm)
        res = Gm_out + Ga

        gamp = self.upsample2(self.glbamp(res)) / torch.sqrt(torch.tensor(self.channel, dtype=torch.float32))
        cat = torch.cat((Ga, Gm_out, gamp), dim=1)
        cat = self.conv_cat(cat)
        base_feat = res + cat

        # === 新增特征融合 ===
        # 拼接基础特征和动态特征
        feat_cat = torch.cat([base_feat, dynamic_feat], dim=1)  # [B, 2C, H, W]

        # 生成注意力权重
        fusion_weight = self.fusion_att(feat_cat)  # [B, 1, H, W]

        # 自适应加权融合
        final_feat = fusion_weight * base_feat + (1 - fusion_weight) * dynamic_feat

        return final_feat



class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decoder, self).__init__()
        self.DWConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.BConv = BConv(out_channel * 2, out_channel, 3, 1, 1)

    def forward(self, x):
        x_left = self.conv(x)
        x_right = self.DWConv(x)
        out = self.BConv(torch.cat((x_left, x_right), dim=1))

        return out



class ExpertModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpertModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def balance_loss(gating_weights):
    # 计算每个专家的平均使用率
    expert_usage = torch.mean(gating_weights, dim=(0, 2, 3))
    # 计算平衡损失
    loss = torch.var(expert_usage)
    return loss





class HierarchyHead(nn.Module):
    def __init__(self, in_channels, level='deep'):
        super().__init__()
        # 根据层级深度设计不同的结构
        if level == 'deep':  # 深层（低分辨率，语义强）
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 1, 1)
            )
        elif level == 'mid':  # 中层（平衡语义与细节）
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 1, 1)
            )
        else:  # 浅层（高分辨率，细节丰富）
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, padding=2),  # 更大卷积核捕捉细节
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 1, 1),
                EdgeAwareModule()  # 添加边缘感知模块
            )

    def forward(self, x):
        return self.conv(x)


class EdgeAwareModule(nn.Module):
    """ 浅层专用的边缘感知模块 """

    def __init__(self):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        return x * (1 + edge)  # 增强边缘区域响应


class pvt(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, num_experts=3):
        super(pvt, self).__init__()

        # encoder
        self.rgb_pvt = mobile_vit_small()
        self.depth_pvt = mobile_vit_small()

        self.fusion1 = DynamicFusion(160, 12, 12)
        self.fusion2 = DynamicFusion(128, 24, 24)
        self.fusion3 = DynamicFusion(96, 48, 48)
        self.fusion4 = DynamicFusion(64, 96, 96)

        # experts
        self.experts1 = nn.ModuleList([ExpertModel(160, 160) for _ in range(num_experts)])
        self.experts2 = nn.ModuleList([ExpertModel(128, 128) for _ in range(num_experts)])
        self.experts3 = nn.ModuleList([ExpertModel(96, 96) for _ in range(num_experts)])
        self.experts4 = nn.ModuleList([ExpertModel(64, 64) for _ in range(num_experts)])
        self.gating_network1 = nn.Sequential(
            nn.Conv2d(160, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.gating_network2 = nn.Sequential(
            nn.Conv2d(128, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.gating_network3 = nn.Sequential(
            nn.Conv2d(96, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.gating_network4 = nn.Sequential(
            nn.Conv2d(64, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # decoder
        # 替换为层级感知监督头
        self.head_x4 = HierarchyHead(64, level='deep')  # x4: 最深层次
        self.head_x3 = HierarchyHead(64, level='mid')  # x3: 中间层次
        self.head_x2 = HierarchyHead(64, level='mid')  # x2: 中间层次
        self.head_x1 = HierarchyHead(64, level='shallow')  # x1: 最浅层次

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder_12 = decoder(160, 64)
        self.sh_attention_12 = sa_layer(64)

        self.decoder_24 = decoder(128, 64)
        self.sh_attention_24 = sa_layer(64)
        self.bconv24 = BConv(128, 64, 3, 1, 1)

        self.decoder_48 = decoder(96, 64)
        self.sh_attention_48 = sa_layer(64)
        self.bconv48 = BConv(128, 64, 3, 1, 1)

        self.decoder_96 = decoder(64, 64)
        self.sh_attention_96 = sa_layer(64)
        self.bconv96 = BConv(128, 64, 3, 1, 1)

    def forward(self, rgb, depth, return_gating_weights=False):
        rgb_list = self.rgb_pvt(rgb)
        depth_list = self.depth_pvt(depth)

        r4 = rgb_list[1]
        r3 = rgb_list[2]
        r2 = rgb_list[3]
        r1 = rgb_list[4]

        d4 = depth_list[1]
        d3 = depth_list[2]
        d2 = depth_list[3]
        d1 = depth_list[4]

        # Encoder
        F1 = self.fusion1(r1, d1)
        F2 = self.fusion2(r2, d2)
        F3 = self.fusion3(r3, d3)
        F4 = self.fusion4(r4, d4)

        # Experts for F1
        gating_weights1 = self.gating_network1(F1)
        expert_outputs1 = [expert(F1) for expert in self.experts1]
        expert_outputs1 = torch.stack(expert_outputs1, dim=1)
        F1 = torch.sum(gating_weights1.unsqueeze(2) * expert_outputs1, dim=1)
        balance_loss1 = balance_loss(gating_weights1)

        # Experts for F2
        gating_weights2 = self.gating_network2(F2)
        expert_outputs2 = [expert(F2) for expert in self.experts2]
        expert_outputs2 = torch.stack(expert_outputs2, dim=1)
        F2 = torch.sum(gating_weights2.unsqueeze(2) * expert_outputs2, dim=1)
        balance_loss2 = balance_loss(gating_weights2)

        # Experts for F3
        gating_weights3 = self.gating_network3(F3)
        expert_outputs3 = [expert(F3) for expert in self.experts3]
        expert_outputs3 = torch.stack(expert_outputs3, dim=1)
        F3 = torch.sum(gating_weights3.unsqueeze(2) * expert_outputs3, dim=1)
        balance_loss3 = balance_loss(gating_weights3)

        # Experts for F4
        gating_weights4 = self.gating_network4(F4)
        expert_outputs4 = [expert(F4) for expert in self.experts4]
        expert_outputs4 = torch.stack(expert_outputs4, dim=1)
        F4 = torch.sum(gating_weights4.unsqueeze(2) * expert_outputs4, dim=1)
        balance_loss4 = balance_loss(gating_weights4)

        # Decoder
        x1_decoder = self.decoder_12(F1)
        x1 = self.sh_attention_12(x1_decoder)

        x2_decoder = self.decoder_24(F2)
        x2_att = self.sh_attention_24(x2_decoder)
        x2_mul = self.upsample2(x1_decoder) * x2_att
        x2_cat = self.bconv24(torch.cat((x2_mul, x2_att), dim=1))
        x2 = x2_cat + self.upsample2(x1)

        x3_decoder = self.decoder_48(F3)
        x3_att = self.sh_attention_48(x3_decoder)
        x3_mul = self.upsample2(x2_decoder) * x3_att
        x3_cat = self.bconv48(torch.cat((x3_mul, x3_att), dim=1))
        x3 = x3_cat + self.upsample2(x2)

        x4_decoder = self.decoder_96(F4)
        x4_att = self.sh_attention_96(x4_decoder)
        x4_mul = self.upsample2(x3_decoder) * x4_att
        x4_cat = self.bconv96(torch.cat((x4_mul, x4_att), dim=1))
        x4 = x4_cat + self.upsample2(x3)

        shape = rgb.size()[2:]  # shape:(384,384)

        # 修改为层级感知输出
        out1 = F.interpolate(self.head_x1(x1), size=shape, mode='bilinear')
        out2 = F.interpolate(self.head_x2(x2), size=shape, mode='bilinear')
        out3 = F.interpolate(self.head_x3(x3), size=shape, mode='bilinear')
        out4 = F.interpolate(self.head_x4(x4), size=shape, mode='bilinear')


        if return_gating_weights:
            # 返回输出和平衡损失
            balance_loss_total = balance_loss1 + balance_loss2 + balance_loss3 + balance_loss4
            return out1, out2, out3, out4, balance_loss_total
        else:
            return out1, out2, out3, out4





        # ###################################################   end   #########################################

    def load_pre(self, pre_model_rgb, pre_model_depth):
        self.rgb_pvt.load_state_dict(torch.load(pre_model_rgb), strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model_rgb}")
        self.depth_pvt.load_state_dict(torch.load(pre_model_depth), strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model_depth}")

    # def load_pre(self, model):
    #     self.edge_layer.load_state_dict(model,strict=False)

# ########################################### end #####################################################
import torch
from torchvision.models import resnet18
from thop import profile

model = pvt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input1 = torch.randn(1, 3, 384, 384)
input1 = input1.to(device)
input2 = torch.randn(1, 3, 384, 384)
input2 = input2.to(device)
flops, params = profile(model.cuda(), inputs=(input1, input2,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))

import torch


iterations = 300   # 重复计算的轮次

model = pvt()
device = torch.device("cuda:0")
model.to(device)

random_input1 = torch.randn(1, 3, 384, 384).to(device)
random_input2 = torch.randn(1, 3, 384, 384).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input1,random_input2)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input1,random_input2)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
