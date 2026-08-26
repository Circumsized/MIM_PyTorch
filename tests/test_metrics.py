import torch

from metrics import batch_mse, batch_psnr, batch_ssim, batch_mae, csi_score


def test_psnr_identical_images():
    x = torch.rand(2, 5, 1, 32, 32)
    psnr = batch_psnr(x, x)
    assert psnr > 90


def test_ssim_identical_images():
    x = torch.rand(2, 5, 1, 32, 32)
    ssim = batch_ssim(x, x)
    assert ssim > 0.99


def test_csi_all_hits():
    y = torch.ones(2, 5, 1, 32, 32) * 0.8
    csi = csi_score(y, y, threshold=0.5)
    assert abs(csi - 1.0) < 1e-6


def test_csi_all_false_alarms():
    gen = torch.ones(2, 5, 1, 32, 32) * 0.8
    gt = torch.zeros(2, 5, 1, 32, 32)
    csi = csi_score(gen, gt, threshold=0.5)
    assert abs(csi) < 1e-6


def test_mse_mae_non_negative():
    x = torch.rand(2, 5, 1, 32, 32)
    y = torch.rand(2, 5, 1, 32, 32)
    assert batch_mse(x, y) >= 0
    assert batch_mae(x, y) >= 0
