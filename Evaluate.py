import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict, defaultdict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
# from sklearn.metrics import roc_auc_score, roc_curve  # 这里加了 roc_curve
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from utils import *
import random
import glob

import argparse
import timm
from model.localizer_cnn import CNNFeatureExtractor, LocalKNN

"""
Evaluate_Corridors_modified.py

This version integrates training-free local anomaly localization using
DINOv2 features + kNN memory (PatchCore-style) with multi-scale fusion,
TTA, and robust image-level pooling (topk+max, quantile, LSE). It also
adds optional per-video normalization and light temporal smoothing for
stability on corridor hazards like water puddles, cables, cellophane,
and screws, commonly seen in Hazards&Robots Corridors.
"""

# ====== 新增：只改热图叠加的实现，其它逻辑不动 ======
def overlay_anomaly_map(amap, image, alpha=0.8, gamma=2.0,
                        colormap=cv2.COLORMAP_JET):
    """
    例图风格：热图叠加在原图上，保留原图纹理 + 用伪彩突出异常区域
      - amap: (H, W) 浮点，值越大越“异常”
      - image: 原图 (H,W,3) 或 (3,H,W) 或 (H,W)；支持 float/uint8
      - alpha: 热图强度（越大越“红”，纹理越不明显），典型 0.4~0.7
      - gamma: 对归一化后的热图做 gamma 校正，让高分区域更突出（典型 1~3）
      - 返回: (H, W, 3) BGR uint8，可直接 cv2.imwrite（叠加后的可视化图）
    """
    if amap is None:
        return None

    # ---------- 0) amap 防御 ----------
    amap = np.asarray(amap, dtype=np.float32)
    amap = np.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- 1) 鲁棒归一化到 [0,1]（分位数抑制极端值） ----------
    lo = float(np.percentile(amap, 1.0))
    hi = float(np.percentile(amap, 99.0))
    if hi > lo:
        amap_norm = (amap - lo) / (hi - lo)
    else:
        a_min = float(amap.min())
        a_max = float(amap.max())
        if a_max > a_min:
            amap_norm = (amap - a_min) / (a_max - a_min)
        else:
            amap_norm = np.zeros_like(amap, dtype=np.float32)
    amap_norm = np.clip(amap_norm, 0.0, 1.0)

    # ---------- 2) gamma 校正（让异常更“亮/红”） ----------
    if gamma is not None and gamma > 0:
        amap_norm = amap_norm ** float(gamma)

    # ---------- 3) 轻微平滑（只影响可视化连续性） ----------
    H, W = amap_norm.shape[:2]
    k = 7 if min(H, W) >= 32 else 0
    if k and k > 1 and k % 2 == 1:
        amap_norm = cv2.GaussianBlur(amap_norm, (k, k), 0)

    # ---------- 4) 伪彩色热图（BGR） ----------
    heat_uint8 = (amap_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)  # (H,W,3) BGR

    # ---------- 5) 准备原图（统一成 HWC, uint8, 3通道） ----------
    if image is None:
        return heat_color

    img = np.asarray(image)

    # 兼容 CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # 灰度图 -> 3通道
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # 尺寸不一致则 resize 到 amap 的尺寸（避免 addWeighted 报错）
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    # dtype/范围统一到 uint8 [0,255]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        # 兼容 [-1,1] / [0,1] / [0,255]
        vmin, vmax = float(img.min()), float(img.max())
        if vmax <= 1.0 and vmin >= 0.0:
            img = img * 255.0
        elif vmin >= -1.0 and vmax <= 1.0:
            img = (img + 1.0) * 0.5 * 255.0
        # 否则默认它已经是接近 0~255 的量级
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        # uint8 也保证一下形状
        img = img.copy()

    # 到这里：img 是 (H,W,3) uint8；heat_color 是 (H,W,3) uint8

    # ---------- 6) 线性融合：保留纹理 + 高亮异常 ----------
    a = float(alpha)
    if a < 0.0: a = 0.0
    if a > 1.0: a = 1.0
    overlay = cv2.addWeighted(img, 1.0 - a, heat_color, a, 0.0)

    return overlay

# ====== 热图部分就改这里，下面保持不动 ======



parser = argparse.ArgumentParser(description="MNAD + Localizer (Corridors)")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anomaly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score (PSNR vs feature-distance)')
parser.add_argument('--th', type=float, default=-1, help='threshold for test updating (set -1 to disable update)')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai, Corridors128, ...')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model (pth)')
parser.add_argument('--m_items_dir', type=str, help='directory of memory keys (pt)')

# ---- Localizer 分支参数 ----
parser.add_argument('--enable_localizer', action='store_true', help='use localizer for pixel heatmaps')
parser.add_argument('--localizer_bank', type=str, default=None, help='path to localizer_bank.pt')
parser.add_argument('--local_backbone', type=str, default='resnet50', help='timm backbone name (e.g., vit_small_patch14_dinov2.lvd142m)')
parser.add_argument('--alpha_img', type=float, default=0.1, help='weight for image-local anomaly score in late fusion')
# ---- 只保存高分帧（新增） ----
parser.add_argument('--save_high_frames', action='store_true', help='save overlay images for high-score frames')
parser.add_argument('--save_threshold', type=float, default=0, help='threshold in [0,1] to save frames')
parser.add_argument('--save_dir', type=str, default=None, help='output dir for overlays; default exp/<ds>/<method>/eval_highframes')
# 默认值改为 50（仅影响保存热图功能；其余逻辑不变）
parser.add_argument('--max_save_per_video', type=int, default=50, help='max number of saved frames per video')
parser.add_argument('--jpeg_quality', type=int, default=85, help='JPEG quality (0-100) for saved overlays')
# ------- 图像级分数聚合参数 -------
parser.add_argument('--image_pooling', type=str, default='topk+max',
                    choices=['mean','max','topk','topk+max','quantile','lse'],
                    help='如何把像素级热图聚合成图像级分数')
parser.add_argument('--image_pooling_p', type=float, default=0.2,
                    help='topk: 取前 p 比例像素；quantile: 使用 1-p 分位数')
parser.add_argument('--image_pooling_blend_max', type=float, default=0.8,
                    help='仅当 image_pooling=topk+max 时，与最大值的线性混合权重')
parser.add_argument('--local_ms_scales', type=str, default='1.0,0.75,0.5',
                    help='逗号分隔的多尺度因子，如 "1.0,0.75,0.5"')
parser.add_argument('--local_gaussian_ksize', type=int, default=3,
                    help='高斯平滑可视化热图的核大小(奇数, 0 表示关闭)')
# ------- 新增的两项实用增强 -------
parser.add_argument('--local_norm', action='store_true',
                    help='在送入 KNN 前对特征做 L2 归一化')
parser.add_argument('--local_tta_hflip', action='store_true',
                    help='Localizer 使用水平翻转 TTA（原图/翻转图热图逐像素取 max）')
# ------- 新增：局部分支的归一化与时间平滑 -------
parser.add_argument('--per_video_norm', action='store_true',
                    help='对局部分支的图像级分数在每个视频内做 min-max 归一化（更稳）')
parser.add_argument('--temporal_ema', type=float, default=0.9,
                    help='对最终分数做指数滑动平均，范围[0,1)，0 表示关闭；建议 0.9 左右')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

# Loading dataset
test_dataset = DataLoader(
    test_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1
)

test_size = len(test_dataset)
test_batch = data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.num_workers_test,
    drop_last=False
)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
assert args.model_dir is not None, "--model_dir is required"
assert args.m_items_dir is not None, "--m_items_dir is required"
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)

# === Localizer init ===
use_local = args.enable_localizer and (args.localizer_bank is not None) and os.path.exists(args.localizer_bank)
if use_local:
    bank = torch.load(args.localizer_bank, map_location='cuda')
    backbone_name = bank.get('backbone', args.local_backbone)
    extractor = CNNFeatureExtractor(backbone_name=backbone_name).cuda().eval()
    knn = LocalKNN(max_items=int(bank['keys'].shape[0])).cuda()
    # 还原 bank
    knn.keys = bank['keys'].cuda().float()
    if 'mean' in bank and 'std' in bank:
        knn.mean = bank['mean'].cuda().float()
        knn.std = bank['std'].cuda().float()

# === 可视化保存根目录 & Top-K 缓存（仅在需要保存时启用） ===
save_high = use_local and args.save_high_frames
if save_high:
    vis_root = args.save_dir or os.path.join('./exp', args.dataset_type, args.method, 'eval_highframes')
    os.makedirs(vis_root, exist_ok=True)
    saved_counts = defaultdict(int)  # 每个视频已保存数（统计用）
    import heapq
    topk_buffers = defaultdict(list)  # 每个视频的最小堆：[(score, global_idx, overlay_bgr), ...]

labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')

# Collect video info
videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

# Prepare containers
labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}
image_score_list = {}   # 存每帧的局部分支图像级分数

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        labels_list = np.append(labels_list, labels[0][4 + label_length: videos[video_name]['length'] + label_length])
    else:
        labels_list = np.append(labels_list, labels[0][label_length: videos[video_name]['length'] + label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []
    image_score_list[video_name] = []  # 初始化

# Reset counters for streaming over dataloader
label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

# Optional temporal EMA state
ema_state = None
ema_gamma = float(args.temporal_ema)
use_ema = (ema_gamma > 0.0 and ema_gamma < 1.0)

for k, (imgs) in enumerate(test_batch):

    # handle video switch
    if args.method == 'pred':
        if k == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = \
            model.forward(imgs[:, 0:3 * 4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
        mse_feas = compactness_loss.item()
        # test-time thresholding
        point_sc = point_score(outputs, imgs[:, 3 * 4:])
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = \
            model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)).item()
        mse_feas = compactness_loss.item()
        # test-time thresholding
        point_sc = point_score(outputs, imgs)

    # memory update (disable if th < 0)
    if args.th >= 0 and point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b x h x w x d
        m_items_test = model.memory.update(query, m_items_test, False)

    # === Localizer: 像素热图 + 图像级分数（多尺度 + 可配置汇聚 + 可选归一化/TTA） ===
    if use_local:
        if args.method == 'pred':
            cur = (imgs[:, 3 * 4: 3 * 4 + 3] + 1) / 2.0  # 当前帧，[0,1]
        else:
            cur = (imgs[:, :3] + 1) / 2.0

        with torch.no_grad():
            # 1) 多尺度热图
            scales = [float(s) for s in (args.local_ms_scales.split(',') if args.local_ms_scales else ['1.0'])]
            amap_pyramid = []
            for s in scales:
                if abs(s - 1.0) < 1e-6:
                    cur_s = cur
                else:
                    # 双线性缩放到较小分辨率再送特征提取
                    h_s = max(8, int(args.h * s))
                    w_s = max(8, int(args.w * s))
                    cur_s = F.interpolate(cur, size=(h_s, w_s), mode='bilinear', align_corners=False)

                # 原图分支
                fmap = extractor(cur_s)  # [B,C,H',W']
                if args.local_norm:
                    fmap = F.normalize(fmap, dim=1)
                amap = knn.anomaly_map(fmap)  # [B,H',W'] ∈ [0,1]
                amap_up = F.interpolate(amap.unsqueeze(1), size=(args.h, args.w),
                                        mode='bilinear', align_corners=False).squeeze(1)  # [B,H,W]

                # 可选：hflip TTA 分支（翻转->特征->热图->再翻回）
                if args.local_tta_hflip:
                    cur_s_flip = torch.flip(cur_s, dims=[-1])
                    fmap_f = extractor(cur_s_flip)
                    if args.local_norm:
                        fmap_f = F.normalize(fmap_f, dim=1)
                    amap_f = knn.anomaly_map(fmap_f)
                    amap_f_up = F.interpolate(amap_f.unsqueeze(1), size=(args.h, args.w),
                                              mode='bilinear', align_corners=False).squeeze(1)
                    # 把翻回后的热图与原图热图逐像素取 max
                    amap_up = torch.maximum(amap_up, amap_f_up.flip(dims=[-1]))

                amap_pyramid.append(amap_up)

            # 融合（逐像素 max 更保守，有利于小而亮的异常）
            amap_up_fused = torch.amax(torch.stack(amap_pyramid, dim=0), dim=0)  # [B,H,W], [0,1]

            # 2) 图像级分数聚合
            B = amap_up_fused.shape[0]
            flat = amap_up_fused.view(B, -1)

            def topk_mean(x, frac):
                k = max(1, int(x.size(1) * frac))
                return torch.topk(x, k, dim=1).values.mean(dim=1)

            if args.image_pooling == 'mean':
                img_score = amap_up_fused.mean(dim=(1, 2))
            elif args.image_pooling == 'max':
                img_score = flat.max(dim=1).values
            elif args.image_pooling == 'topk':
                img_score = topk_mean(flat, args.image_pooling_p)
            elif args.image_pooling == 'topk+max':
                tk = topk_mean(flat, args.image_pooling_p)
                mx = flat.max(dim=1).values
                w = float(args.image_pooling_blend_max)
                img_score = (1.0 - w) * tk + w * mx
            elif args.image_pooling == 'quantile':
                # 使用 1-p 分位数（如 p=0.05 -> 95%分位）
                q = torch.quantile(flat, q=min(max(1.0 - args.image_pooling_p, 0.0), 1.0), dim=1)
                img_score = q
            elif args.image_pooling == 'lse':
                N = flat.size(1)
                img_score = torch.logsumexp(flat, dim=1) - math.log(N)
            else:
                img_score = amap_up_fused.mean(dim=(1, 2))  # fallback

            # 兼容 test_batch_size=1
            img_score_val = float(img_score[0].item()) if img_score.ndim > 0 else float(img_score.item())

            # 3) 可选：对保存可视化用热图做轻微平滑（不影响分数）
            if save_high and args.local_gaussian_ksize and args.local_gaussian_ksize > 1 and args.local_gaussian_ksize % 2 == 1:
                # 仅用于保存叠加图时的视觉平滑；分数仍然来自 amap_up_fused
                pass
    else:
        img_score_val = 0.0

    # accumulate metrics per video
    cur_video_name = videos_list[video_num].split('/')[-1]
    psnr_list[cur_video_name].append(psnr(mse_imgs))
    feature_distance_list[cur_video_name].append(mse_feas)
    image_score_list[cur_video_name].append(img_score_val)

    # === 改动点：缓存“高分帧”用于每视频 Top-K 选择（不立即落盘） ===
    if use_local and save_high:
        # 生成叠加图（BGR uint8）
        amap_np = amap_up_fused[0].detach().float().cpu().numpy() if use_local else None
        cur_np = cur[0].detach().float().cpu().numpy() if use_local else None
        if use_local and args.local_gaussian_ksize and args.local_gaussian_ksize > 1 and args.local_gaussian_ksize % 2 == 1:
            # 高斯平滑只影响可视化
            amap_np = cv2.GaussianBlur(amap_np, (args.local_gaussian_ksize, args.local_gaussian_ksize), 0)

        # ★ 这里还是调用 overlay_anomaly_map，但实现已经换成新的那版
        overlay = overlay_anomaly_map(amap_np, cur_np, alpha=0.5) if use_local else None

        # 仅当分数 >= 阈值时才进入 Top-K（若想总是保满K帧，可把 --save_threshold 设为 0）
        if img_score_val >= args.save_threshold:
            buf = topk_buffers[cur_video_name]
            item = (float(img_score_val), int(k), overlay)  # (分数, 全局帧号k, 叠加图)
            if len(buf) < int(args.max_save_per_video):
                heapq.heappush(buf, item)            # 维护最小堆
            else:
                heapq.heappushpop(buf, item)         # 只保留 Top-K

# === 改动点：将每个视频的 Top-K 叠加图一次性写盘（得分从大到小），不影响 AUC ===
if use_local and save_high:
    from heapq import nlargest
    for vname, heap in topk_buffers.items():
        if not heap:
            continue
        out_dir = os.path.join(vis_root, vname)
        os.makedirs(out_dir, exist_ok=True)
        # 由高到低保存；文件名仍用全局帧索引k，保持原有命名风格
        for score, idx, overlay in nlargest(int(args.max_save_per_video), heap):
            out_path = os.path.join(out_dir, f"{idx:06d}.jpg")
            try:
                cv2.imwrite(out_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            except Exception:
                cv2.imwrite(out_path, overlay)
            saved_counts[vname] += 1

# ---- Measuring the abnormality score and the AUC ----
# 原两路：PSNR & feature distance
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    base_scores = score_sum(
        anomaly_score_list(psnr_list[video_name]),
        anomaly_score_list_inv(feature_distance_list[video_name]),
        args.alpha
    )
    anomaly_score_total_list += base_scores

anomaly_score_total_list = np.asarray(anomaly_score_total_list, dtype=np.float32)

# 第三路：局部分支（图像级分数）—— 归一化后线性融合
if use_local:
    image_score_total_list = []
    if args.per_video_norm:
        # 每个视频内做 min-max 归一化，可减少跨视频的亮度/背景差异
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            arr = np.asarray(image_score_list[video_name], dtype=np.float32)
            if arr.size > 0:
                vmin, vmax = float(np.min(arr)), float(np.max(arr))
                if vmax > vmin:
                    arr = (arr - vmin) / (vmax - vmin)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
            image_score_total_list += arr.tolist()
        image_score_total_list = np.asarray(image_score_total_list, dtype=np.float32)
    else:
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            image_score_total_list += image_score_list[video_name]
        image_score_total_list = np.asarray(image_score_total_list, dtype=np.float32)
        if image_score_total_list.size > 0:
            img_min, img_max = float(image_score_total_list.min()), float(image_score_total_list.max())
            if img_max > img_min:
                image_score_total_list = (image_score_total_list - img_min) / (img_max - img_min)
            else:
                image_score_total_list = np.zeros_like(image_score_total_list, dtype=np.float32)

    # 融合
    anomaly_score_total_list = (1 - args.alpha_img) * anomaly_score_total_list + \
                               args.alpha_img * image_score_total_list

# 轻量时间平滑（可选）
if use_ema:
    ema = None
    out = []
    for s in anomaly_score_total_list.tolist():
        ema = s if ema is None else (ema_gamma * ema + (1.0 - ema_gamma) * s)
        out.append(ema)
    anomaly_score_total_list = np.asarray(out, dtype=np.float32)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy * 100, '%')

# ====== 新增：绘制整体 ROC 曲线（与上面 AUC 调用保持一致） ======
y_true = 1 - labels_list            # 上面 AUC 里正类是“正常”，这里保持一致
y_score = anomaly_score_total_list  # 分数越大越“正常”
# ====== 新增：AP(AUPR) + Best-F1（不影响原有AUC/ROC/画图） ======
ap = average_precision_score(y_true, y_score)
print('AP (AUPR): ', ap * 100, '%')

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

# 注意：sklearn 的 precision/recall 比 thresholds 多 1 个点，最后一个点没有对应阈值 :contentReference[oaicite:3]{index=3}
if pr_thresholds is not None and len(pr_thresholds) > 0:
    precision_t = precision[1:]
    recall_t = recall[1:]
    f1 = 2.0 * precision_t * recall_t / (precision_t + recall_t + 1e-12)
    best_idx = int(np.argmax(f1))
    print('Best-F1: ', float(f1[best_idx]) * 100, '%  @thr=', float(pr_thresholds[best_idx]))
else:
    print('Best-F1: N/A (no valid thresholds)')
# ============================================================

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)
# ====== 新增：导出 ROC 相关数组，供多模型叠加画图用 ======
export_root = os.path.join('./roc_exports', args.dataset_type)
os.makedirs(export_root, exist_ok=True)

# 用 checkpoint 文件名当作模型标识（避免你四个模型导出时互相覆盖）
model_tag = os.path.splitext(os.path.basename(args.model_dir))[0]

export_path = os.path.join(export_root, f"roc_{model_tag}.npz")

np.savez(
    export_path,
    # 画 ROC 必备
    y_true=np.asarray(y_true, dtype=np.uint8),
    y_score=np.asarray(y_score, dtype=np.float32),
    fpr=np.asarray(fpr, dtype=np.float32),
    tpr=np.asarray(tpr, dtype=np.float32),
    thresholds=np.asarray(thresholds, dtype=np.float32),
    auc=np.asarray([roc_auc], dtype=np.float32),

    # （可选但很有用）记录一些元信息，后面检查一致性
    dataset_type=np.asarray([args.dataset_type]),
    method=np.asarray([args.method]),
    model_dir=np.asarray([args.model_dir]),
)

print("[ROC-EXPORT] Saved:", export_path)
# ===========================================================

roc_root = os.path.join('./anomaly_plots', args.dataset_type)
os.makedirs(roc_root, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
pad_x = 0.05
pad_y = 0.05
plt.xlim(-pad_x, 1.0 + pad_x)
plt.ylim(-pad_y, 1.0 + pad_y)
plt.xlabel('False Positive Rate (FRR)')
plt.ylabel('True Positive Rate (TRR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_path = os.path.join(roc_root, f'ROC_{args.dataset_type}.png')
plt.savefig(roc_path, dpi=300)
plt.close()

# ===================== 下面开始是新增的“画图”代码 前400帧=====================

video_scores = OrderedDict()
start_idx = 0
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    v_len = videos[video_name]['length']
    if args.method == 'pred':
        num = v_len - 4  # 前 4 帧没有分数
    else:
        num = v_len
    end_idx = start_idx + num
    video_scores[video_name] = anomaly_score_total_list[start_idx:end_idx]
    start_idx = end_idx

# 同样方式，把标签按视频切分（和上面 labels_list 逻辑保持一致）
video_labels = OrderedDict()
label_offset = 0
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    v_len = videos[video_name]['length']
    if args.method == 'pred':
        s = label_offset + 4
        e = label_offset + v_len
    else:
        s = label_offset
        e = label_offset + v_len
    video_labels[video_name] = labels[0][s:e]
    label_offset += v_len

# 小工具：把 0/1 标签转成若干个 [start, end) 区间，用来高亮异常区域
def _ranges_from_labels(label_1d, anomaly_value=1):
    arr = np.asarray(label_1d).astype(int)
    ranges = []
    in_range = False
    start = 0
    for i, lb in enumerate(arr):
        if lb == anomaly_value and not in_range:
            in_range = True
            start = i
        elif lb != anomaly_value and in_range:
            in_range = False
            ranges.append((start, i))
    if in_range:
        ranges.append((start, len(arr)))
    return ranges

# 输出目录：./anomaly_plots/<dataset_type>/
plot_root = os.path.join('./anomaly_plots', args.dataset_type)
os.makedirs(plot_root, exist_ok=True)

# ===== 这里改成可以自己调帧数区间 =====
FRAME_START = 8380    # 起始帧下标（含）
FRAME_END = 8980    # 结束帧下标（不含），设为 None 表示一直到最后一帧
# =================================

for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    scores_full = np.asarray(video_scores[video_name], dtype=np.float32)
    labels_full = np.asarray(video_labels[video_name], dtype=np.int32)

    if scores_full.size == 0:
        continue

    # 计算实际使用的起止下标（防止越界）
    start = max(0, FRAME_START)
    if FRAME_END is None:
        end = len(scores_full)
    else:
        end = min(len(scores_full), FRAME_END)

    if end <= start:
        # 区间非法或没有帧可画，跳过当前视频
        continue

    # 只取 [FRAME_START, FRAME_END) 这个区间
    scores = scores_full[start:end]
    labels_v = labels_full[start:end]

    # 当前 anomaly_score_total_list 越大越“正常”，
    # 为了画图时“越大越异常”，这里取反再归一化
    scores_for_plot = -scores

    # 归一化到 [0,1]，更好看
    s_min, s_max = float(scores_for_plot.min()), float(scores_for_plot.max())
    if s_max > s_min:
        y = (scores_for_plot - s_min) / (s_max - s_min)
    else:
        y = np.zeros_like(scores_for_plot)

    x = np.arange(FRAME_START,FRAME_END)  # 这里保留原逻辑，用子区间内的相对帧号

    fig, ax = plt.subplots(figsize=(6, 4))

    # 只画一条橙色曲线
    ax.plot(x, y, lw=1.5, color='tab:orange', label='anomaly score')

    # 高亮标签为 1 的异常帧区间（蓝色背景）
    for st, ed in _ranges_from_labels(labels_v, anomaly_value=1):
        ax.axvspan(FRAME_START+st, FRAME_START+ed, color='skyblue', alpha=0.3)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Anomaly score')
    ax.set_title(f'{args.dataset_type}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(
        plot_root,
        f'{video_name}_anomaly_curve_frames{start}-{end}.png'
    )
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
