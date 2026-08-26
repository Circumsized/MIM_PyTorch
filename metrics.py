import numpy as np
import torch


def batch_mse(gen_frames, gt_frames):
    x = gen_frames.float()
    y = gt_frames.float()
    mse = torch.mean((x - y) ** 2)
    return mse.item()


def batch_psnr(gen_frames, gt_frames):
    x = gen_frames.float().clamp(0, 1)
    y = gt_frames.float().clamp(0, 1)
    reduce_dims = list(range(1, x.dim()))
    mse = torch.mean((x - y) ** 2, dim=reduce_dims)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def batch_ssim(gen_frames, gt_frames):
    x = gen_frames.float().clamp(0, 1)
    y = gt_frames.float().clamp(0, 1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    reduce_dims = list(range(1, x.dim()))
    mu_x = x.mean(dim=reduce_dims, keepdim=True)
    mu_y = y.mean(dim=reduce_dims, keepdim=True)
    sigma_x = x.var(dim=reduce_dims, keepdim=True, unbiased=False)
    sigma_y = y.var(dim=reduce_dims, keepdim=True, unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=reduce_dims, keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean().item()


def batch_mae(gen_frames, gt_frames):
    x = gen_frames.float()
    y = gt_frames.float()
    mae = torch.mean(torch.abs(x - y))
    return mae.item()


def csi_score(gen_frames, gt_frames, threshold=0.5):
    x = (gen_frames.float().clamp(0, 1) > threshold).float()
    y = (gt_frames.float().clamp(0, 1) > threshold).float()

    hits = (x * y).sum()
    misses = ((1 - x) * y).sum()
    false_alarms = (x * (1 - y)).sum()

    csi = hits / (hits + misses + false_alarms + 1e-10)
    return csi.item()
