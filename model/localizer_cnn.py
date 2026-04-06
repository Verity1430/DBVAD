# model/localizer_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os

class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        out_indices = (-4, -2, -1),
        pretrained: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1),  persistent=False)

        # 本地权重（你已放到 ./opt/weights/...）
        default_local = os.path.abspath("./opt/weights/vit_s14_dinov2/model.safetensors")
        local_file = os.environ.get("TIMM_PRETRAINED_FILE", default_local)
        use_overlay = os.path.exists(local_file)

        model_name = backbone_name
        overlay = {'file': local_file} if use_overlay else {}

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=overlay
            )
        except Exception as e:
            if use_overlay:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False,
                    features_only=True,
                    out_indices=out_indices,
                    checkpoint_path=local_file
                )
            else:
                raise RuntimeError(
                    f"Cannot load local weights for '{model_name}'. "
                    f"File not found: {local_file}. "
                    "Please place DINOv2 weights here or pass --local_backbone local-dir:/abs/path/to/folder"
                ) from e

        try:
            chs = self.backbone.feature_info.channels()
            self.out_dim = sum(chs)
        except Exception:
            self.out_dim = getattr(self.backbone, "num_features", 768)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = (x - self.mean) / self.std

        # ==== 仅修补这里：让 ViT 接受任意尺寸且返回 NHWC 特征 ====
        core = getattr(self.backbone, "model", self.backbone)
        if hasattr(core, "patch_embed"):
            pe = core.patch_embed
            # 允许动态输入尺寸（如果模型实现了这些属性）
            if hasattr(core, "dynamic_img_size"):
                core.dynamic_img_size = True
            if hasattr(core, "dynamic_img_pad"):
                core.dynamic_img_pad = True
            if hasattr(pe, "strict_img_size") and pe.strict_img_size:
                pe.strict_img_size = False
            # 关键：让 PatchEmbed 输出 BHWC 而不是 BNC
            if hasattr(pe, "flatten"):
                pe.flatten = False
            if hasattr(pe, "output_fmt"):
                pe.output_fmt = "NHWC"

            # 若输入尺寸不能整除 patch_size，手动做最小补边（右、下）
            ps = getattr(pe, "patch_size", 14)
            if isinstance(ps, (tuple, list)):
                ph, pw = int(ps[0]), int(ps[1])
            else:
                ph = pw = int(ps)
            H, W = x.shape[-2:]
            pad_h = (ph - (H % ph)) % ph
            pad_w = (pw - (W % pw)) % pw
            if (pad_h | pad_w) != 0:
                # F.pad 的顺序是 (left, right, top, bottom)
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        # =====================================================

        feats = self.backbone(x)  # list of feature maps
        # 统一到最高分辨率
        Ht, Wt = feats[0].shape[-2:]
        up = [F.interpolate(f, size=(Ht, Wt), mode="bilinear", align_corners=False) for f in feats]
        f_cat = torch.cat(up, dim=1)
        f_cat = F.normalize(f_cat, dim=1)
        return f_cat


class LocalKNN(nn.Module):
    def __init__(
        self,
        max_items: int = 80000,
        device: str = "cuda",
        topk: int = 5,
        q_chunk: int = 16384,
        k_chunk: int = 65536
    ):
        super().__init__()
        self.max_items = int(max_items)
        self.topk = int(max(1, topk))
        self.q_chunk = int(q_chunk)
        self.k_chunk = int(k_chunk)
        self.register_buffer("keys", torch.empty(0, 1), persistent=False)

    @torch.no_grad()
    def _init_if_needed(self, dim: int, device="cuda"):
        if self.keys.numel() == 0:
            self.keys = torch.empty(0, dim, device=device)

    @torch.no_grad()
    def enqueue(self, feat_map: torch.Tensor):
        B, C, H, W = feat_map.shape
        self._init_if_needed(C, device=feat_map.device)
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

        remain = self.max_items - self.keys.shape[0]
        if remain <= 0:
            return
        if patches.shape[0] > remain:
            idx = torch.randperm(patches.shape[0], device=patches.device)[:remain]
            patches = patches[idx]

        self.keys = torch.cat([self.keys, patches], dim=0)

    @torch.no_grad()
    def anomaly_map(self, feat_map: torch.Tensor) -> torch.Tensor:
        assert self.keys.numel() > 0, "LocalKNN: memory bank is empty. Please build it first."
        B, C, H, W = feat_map.shape
        device = feat_map.device

        q = feat_map.permute(0, 2, 3, 1).reshape(-1, C).to(device)
        N = q.shape[0]
        M = self.keys.shape[0]

        best = torch.full((N, self.topk), float('inf'), device=device)

        for qs in range(0, N, self.q_chunk):
            q_blk = q[qs: qs + self.q_chunk]
            q2 = (q_blk * q_blk).sum(dim=1, keepdim=True)
            best_blk = torch.full((q_blk.size(0), self.topk), float('inf'), device=device)

            for ks in range(0, M, self.k_chunk):
                k_blk = self.keys[ks: ks + self.k_chunk]
                k2 = (k_blk * k_blk).sum(dim=1)
                sim = q_blk @ k_blk.t()
                dist2 = q2 + k2.unsqueeze(0) - 2 * sim

                cand = torch.cat([best_blk, dist2], dim=1)
                best_blk, _ = torch.topk(cand, k=self.topk, dim=1, largest=False)

            best[qs: qs + self.q_chunk] = best_blk

        topk_mean = best.mean(dim=1).view(B, H, W)
        vmin = topk_mean.amin(dim=(-1, -2), keepdim=True)
        vmax = topk_mean.amax(dim=(-1, -2), keepdim=True)
        amap = (topk_mean - vmin) / (vmax - vmin + 1e-6)
        return amap
