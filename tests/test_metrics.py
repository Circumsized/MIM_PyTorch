import pytest
import torch

from metrics import batch_mae, batch_mse, batch_psnr, batch_ssim, csi_score


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


def test_psnr_rejects_nonpositive_data_range():
    x = torch.rand(2, 5, 1, 32, 32)
    with pytest.raises(ValueError, match="data_range"):
        batch_psnr(x, x, data_range=0)
    with pytest.raises(ValueError, match="data_range"):
        batch_ssim(x, x, data_range=-1)


def test_csi_rejects_threshold_out_of_range():
    x = torch.rand(2, 5, 1, 32, 32)
    with pytest.raises(ValueError, match="threshold"):
        csi_score(x, x, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        csi_score(x, x, threshold=-0.1)


def test_ssim_low_contrast_clamped_to_one():
    # Constant (zero-variance) samples fall below floor_std and must be
    # reported as SSIM 1.0 instead of a C1/C2-dominated garbage value.
    x = torch.zeros(2, 5, 1, 32, 32)
    assert abs(batch_ssim(x, x) - 1.0) < 1e-6
    dark = torch.full((2, 5, 1, 32, 32), 1e-5)
    assert abs(batch_ssim(dark, dark) - 1.0) < 1e-6


def test_empty_batch_rejected():
    empty = torch.zeros(0, 5, 1, 8, 8)
    ok = torch.rand(1, 5, 1, 8, 8)
    for fn in (batch_mse, batch_psnr, batch_ssim, batch_mae, csi_score):
        with pytest.raises(ValueError, match="empty batch"):
            fn(empty, ok)
        with pytest.raises(ValueError, match="empty batch"):
            fn(ok, empty)


def test_csi_respects_data_range():
    # Values in [0, 2]: with data_range=2.0 and threshold=0.5 the cut is
    # at 1.0, so 0.8 is below threshold and 1.6 is above.
    gen = torch.full((1, 1, 1, 4, 4), 1.6)
    gt = torch.full((1, 1, 1, 4, 4), 0.8)
    csi = csi_score(gen, gt, threshold=0.5, data_range=2.0)
    assert abs(csi) < 1e-6  # all false alarms
    csi2 = csi_score(gt, gt, threshold=0.5, data_range=2.0)
    assert abs(csi2) < 1e-6  # nothing above threshold -> no hits, CSI 0
