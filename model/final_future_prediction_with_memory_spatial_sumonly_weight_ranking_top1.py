import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *

# ===== CBAM =====
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享 MLP（1x1 conv 版本，便于保持空间无关）
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        p = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)
    def forward(self, x):
        return self.sa(self.ca(x))  # 顺序：Channel -> Spatial


# ===== 带 CBAM 的 Encoder / Decoder（形状与返回不变） =====
class Encoder(nn.Module):
    def __init__(self, t_length=5, n_channel=3, cbam_reduction=16):
        super().__init__()

        def Basic(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=False),
                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=False),
                CBAM(o, reduction=cbam_reduction)
            )

        def Basic_(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=False),
                nn.Conv2d(o, o, 3, 1, 1),
                CBAM(o, reduction=cbam_reduction)
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = nn.MaxPool2d(2, 2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = nn.MaxPool2d(2, 2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = nn.MaxPool2d(2, 2)

        self.moduleConv4 = Basic_(256, 512)

        # 与原结构对齐（即使 forward 未用到）
        self.moduleBatchNorm = nn.BatchNorm2d(512)
        self.moduleReLU = nn.ReLU(inplace=False)

    def forward(self, x):
        c1 = self.moduleConv1(x)        # 64
        p1 = self.modulePool1(c1)

        c2 = self.moduleConv2(p1)       # 128
        p2 = self.modulePool2(c2)

        c3 = self.moduleConv3(p2)       # 256
        p3 = self.modulePool3(c3)

        c4 = self.moduleConv4(p3)       # 512

        # 与原版一致的返回：bottleneck, skip1, skip2, skip3
        return c4, c1, c2, c3


class Decoder(nn.Module):
    def __init__(self, t_length=5, n_channel=3, cbam_reduction=16):
        super().__init__()

        def Basic(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=False),
                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=False),
                CBAM(o, reduction=cbam_reduction)
            )

        def Gen(i, o, nc):
            return nn.Sequential(
                nn.Conv2d(i, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.ReLU(inplace=False),
                nn.Conv2d(nc, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.ReLU(inplace=False),
                nn.Conv2d(nc, o, 3, 1, 1), nn.Tanh()
            )

        def Upsample(i, o):
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(o), nn.ReLU(inplace=False)
            )

        self.moduleConv      = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3   = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2   = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1   = Gen(128, n_channel, 64)

    def forward(self, x, skip1, skip2, skip3):
        t = self.moduleConv(x)

        u4 = self.moduleUpsample4(t)
        cat4 = torch.cat((skip3, u4), dim=1)

        d3 = self.moduleDeconv3(cat4)
        u3 = self.moduleUpsample3(d3)
        cat3 = torch.cat((skip2, u3), dim=1)

        d2 = self.moduleDeconv2(cat3)
        u2 = self.moduleUpsample2(d2)
        cat2 = torch.cat((skip1, u2), dim=1)

        out = self.moduleDeconv1(cat2)
        return out


class convAE(nn.Module):
    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512, key_dim=512,
                 temp_update=0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        # 带 CBAM 的编码器/解码器
        self.encoder = Encoder(t_length, n_channel, cbam_reduction=16)
        self.decoder = Decoder(t_length, n_channel, cbam_reduction=16)

        # 原有 Memory 模块保持不变（这里假设你已有实现）
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):
        fea, skip1, skip2, skip3 = self.encoder(x)

        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = \
                self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            # —— 返回项顺序与数量保持不变 ——
            return (output, fea, updated_fea, keys,
                    softmax_score_query, softmax_score_memory,
                    separateness_loss, compactness_loss)
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = \
                self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            # —— 返回项顺序与数量保持不变 ——
            return (output, fea, updated_fea, keys,
                    softmax_score_query, softmax_score_memory,
                    query, top1_keys, keys_ind, compactness_loss)
                                          



    
    
