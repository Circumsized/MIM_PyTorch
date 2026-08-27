import torch


def _guard_nonempty(gen_frames, gt_frames, name):
    if gen_frames.shape[0] == 0 or gt_frames.shape[0] == 0:
        raise ValueError(f"{name}: received empty batch (batch_size=0)")


def batch_mse(gen_frames, gt_frames):
    _guard_nonempty(gen_frames, gt_frames, "batch_mse")
    x = gen_frames.float()
    y = gt_frames.float()
    return torch.mean((x - y) ** 2)


def batch_psnr(gen_frames, gt_frames, data_range=1.0):
    _guard_nonempty(gen_frames, gt_frames, "batch_psnr")
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    x = gen_frames.float().clamp(0, data_range)
    y = gt_frames.float().clamp(0, data_range)
    reduce_dims = list(range(1, x.dim()))
    mse = torch.mean((x - y) ** 2, dim=reduce_dims)
    psnr = 10 * torch.log10(data_range**2 / (mse + 1e-10))
    return psnr.mean()


def batch_ssim(gen_frames, gt_frames, data_range=1.0):
    """Global SSIM (no local windowing) with low-dynamic-range protection.

    When per-sample pixel variance falls below ``floor_std`` the SSIM
    contribution is clamped to 1.0 so that a low-contrast (dark) sample
    does not spuriously inflate or deflate the aggregate metric.
    ``floor_std`` is expressed as a fraction of ``data_range``.
    """
    _guard_nonempty(gen_frames, gt_frames, "batch_ssim")
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    x = gen_frames.float().clamp(0, data_range)
    y = gt_frames.float().clamp(0, data_range)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    floor_std = 1e-3 * data_range

    reduce_dims = list(range(1, x.dim()))
    mu_x = x.mean(dim=reduce_dims, keepdim=True)
    mu_y = y.mean(dim=reduce_dims, keepdim=True)
    sigma_x = x.var(dim=reduce_dims, keepdim=True, unbiased=False)
    sigma_y = y.var(dim=reduce_dims, keepdim=True, unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=reduce_dims, keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )

    low_contrast_mask = (sigma_x.sqrt() < floor_std) & (sigma_y.sqrt() < floor_std)
    ssim_map = ssim_map.masked_fill(low_contrast_mask, 1.0)
    return ssim_map.mean()


def batch_mae(gen_frames, gt_frames):
    _guard_nonempty(gen_frames, gt_frames, "batch_mae")
    x = gen_frames.float()
    y = gt_frames.float()
    return torch.mean(torch.abs(x - y))


def csi_score(gen_frames, gt_frames, threshold=0.5, data_range=1.0):
    """Critical success index at a binary threshold.

    ``threshold`` and ``data_range`` define the pixel range: inputs are
    clamped to [0, data_range] before thresholding. This keeps the binary
    comparison consistent with ``batch_psnr`` / ``batch_ssim``.
    """
    _guard_nonempty(gen_frames, gt_frames, "csi_score")
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    x = (gen_frames.float().clamp(0, data_range) > threshold * data_range).float()
    y = (gt_frames.float().clamp(0, data_range) > threshold * data_range).float()

    hits = (x * y).sum()
    misses = ((1 - x) * y).sum()
    false_alarms = (x * (1 - y)).sum()

    return hits / (hits + misses + false_alarms + 1e-10)
