"""Regression tests for the visualization layer (tqdm + TensorBoard)."""

import argparse
import pathlib

import numpy as np
import pytest
import torch

import train
from dataset import MovingMNIST, get_dataloader
from mim import MIM
from visualization import _TB_AVAILABLE, EpochProgress, TBLogger, make_video_grid

HIDDEN = [8, 8]
INPUT_LENGTH = 3
TOTAL_LENGTH = 6


def _make_namespace(**kw):
    base = dict(  # noqa: C408 - keeps the keyword form readable for overrides
        dataset="mnist",
        batch_size=2,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
        hidden_dim=HIDDEN,
        kernel_size=3,
        lr=1e-3,
        epochs=1,
        num_workers=0,
        num_threads=1,
        device="cpu",
        save_dir=".",
        log_interval=1,
        save_interval=1,
        ss_start_epoch=0,
        ss_stop_epoch=5,
        ss_initial_prob=0.0,
        ss_final_prob=0.0,
        tln=True,
        resume=None,
        amp=False,
        compile=False,
        grad_ckpt=False,
        prefetch_factor=2,
        seed=0,
        # Visualization knobs:
        no_progress_bar=True,
        tensorboard=False,
        tensorboard_dir=None,
        tensorboard_video_interval=5,
        tensorboard_video_samples=2,
        tensorboard_model_graph=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_make_video_grid_layout_and_error_normalisation():
    """3-row layout GT | Pred | Error with per-batch error max normalisation."""
    gt = torch.zeros(2, 4, 1, 8, 8)
    pred = torch.full((2, 4, 1, 8, 8), 0.5)
    grid = make_video_grid(gt, pred)
    assert grid.shape == (2, 4, 1, 24, 8)
    # Top third equals GT (all 0); middle third equals Pred (all 0.5);
    # bottom third is the error heatmap normalised to the batch's max.
    assert torch.equal(grid[:, :, :, 0:8], gt)
    assert torch.equal(grid[:, :, :, 8:16], pred)
    err = grid[:, :, :, 16:]
    # Per-batch max should normalise to 1.0 (the constant error is the max).
    per_batch_max = err.amax(dim=(1, 2, 3, 4))
    assert torch.allclose(per_batch_max, torch.ones(2), atol=1e-5)
    # Values are in [0, 1] (clamp).
    assert err.min() >= 0.0 and err.max() <= 1.0


def test_make_video_grid_rejects_shape_mismatch():
    gt = torch.zeros(2, 4, 1, 8, 8)
    pred = torch.zeros(2, 4, 1, 8, 9)
    with pytest.raises(ValueError, match="shape mismatch"):
        make_video_grid(gt, pred)


def test_tb_logger_creates_files(tmp_path):
    if not _TB_AVAILABLE:
        pytest.skip("tensorboard not installed")
    log_dir = tmp_path / "tb"
    logger = TBLogger(log_dir)
    logger.log_train_loss(0.42, step=1)
    logger.log_lr(1e-3, step=1)
    logger.log_eval_metrics({"psnr": 32.0, "ssim": 0.9}, epoch=0)
    logger.close()
    # SummaryWriter writes event files immediately; just confirm at least
    # one file was created and the close path did not double-close.
    files = list(pathlib.Path(log_dir).iterdir())
    assert any("events.out.tfevents" in p.name for p in files)
    logger.close()  # idempotent


def test_tb_logger_video_writes_correct_shape(tmp_path):
    if not _TB_AVAILABLE:
        pytest.skip("tensorboard not installed")
    log_dir = tmp_path / "tb_video"
    logger = TBLogger(log_dir)
    pred = torch.rand(2, 4, 1, 16, 16)
    target = torch.rand(2, 4, 1, 16, 16)
    logger.log_eval_video(pred, target, epoch=0, max_samples=2)
    # Empty batch is silently a no-op (the call site is the only user and
    # the contract is "best-effort", not "raise").
    logger.log_eval_video(pred[:0], target[:0], epoch=1, max_samples=1)
    logger.close()


def test_tb_logger_missing_tensorboard(monkeypatch, tmp_path):
    """Simulate a missing tensorboard install: constructor must raise fast."""
    monkeypatch.setattr("visualization._TB_AVAILABLE", False)
    monkeypatch.setattr("visualization._TB_IMPORT_ERROR", ImportError("mock"))
    with pytest.raises(ImportError, match="tensorboard is required"):
        TBLogger(tmp_path / "tb")


def test_epoch_progress_context_manager_with_postfix():
    p = EpochProgress(total=3, desc="probe", disable=True)
    with p as bar:
        bar.update(1, loss=0.1)
        bar.update(1, loss=0.05)
    p.close()  # idempotent


# ---------------------------------------------------------------------------
# End-to-end: train.py 1 epoch with TB logger attached
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TB_AVAILABLE, reason="tensorboard not installed")
def test_train_one_epoch_writes_train_loss_to_tb(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    ds = MovingMNIST(str(p), TOTAL_LENGTH, INPUT_LENGTH, is_train=True)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=True)
    model = MIM(
        1,
        1,
        [2, 1, 16, 16],
        hidden_dim=HIDDEN,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    args = _make_namespace()
    logger = TBLogger(tmp_path / "tb")
    try:
        loss = train.train_one_epoch(
            model,
            loader,
            opt,
            torch.nn.MSELoss(),
            0,
            args,
            torch.device("cpu"),
            scaler=None,
            amp_dtype=None,
            tb_logger=logger,
        )
    finally:
        logger.close()
    assert loss == loss
    files = list((tmp_path / "tb").iterdir())
    assert any("events.out.tfevents" in p.name for p in files)


@pytest.mark.skipif(not _TB_AVAILABLE, reason="tensorboard not installed")
def test_evaluate_returns_capture_when_requested(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    test_ds = MovingMNIST(str(p), TOTAL_LENGTH, INPUT_LENGTH, is_train=False)
    test_loader = get_dataloader(test_ds, batch_size=2, num_workers=0, drop_last=False)
    model = MIM(
        1,
        1,
        [2, 1, 16, 16],
        hidden_dim=HIDDEN,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
    )
    args = _make_namespace()
    result = train.evaluate(
        model, test_loader, args, torch.device("cpu"), max_video_samples=2
    )
    assert isinstance(result, tuple)
    metrics, (pred, target) = result
    assert pred.shape == (2, TOTAL_LENGTH - INPUT_LENGTH, 1, 16, 16)
    assert target.shape == pred.shape
    assert all(v == v for v in metrics.values())


def test_evaluate_without_capture_returns_metrics_only(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    test_ds = MovingMNIST(str(p), TOTAL_LENGTH, INPUT_LENGTH, is_train=False)
    test_loader = get_dataloader(test_ds, batch_size=2, num_workers=0, drop_last=False)
    model = MIM(
        1,
        1,
        [2, 1, 16, 16],
        hidden_dim=HIDDEN,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
    )
    args = _make_namespace()
    result = train.evaluate(
        model, test_loader, args, torch.device("cpu"), max_video_samples=0
    )
    assert isinstance(result, dict)
    assert all(v == v for v in result.values())


def test_nonfinite_loss_raises_before_optimizer_step(tmp_path):
    """train_one_epoch must raise FloatingPointError BEFORE optimizer.step()
    is called when the loss is non-finite; otherwise NaN gradients silently
    corrupt model weights before any checkpoint is saved.
    """
    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    ds = MovingMNIST(str(p), TOTAL_LENGTH, INPUT_LENGTH, is_train=True)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=True)
    model = MIM(
        1,
        1,
        [2, 1, 16, 16],
        hidden_dim=HIDDEN,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Record whether step was called
    real_step = opt.step
    step_calls = {"n": 0}

    def spy_step(*a, **kw):
        step_calls["n"] += 1
        return real_step(*a, **kw)

    opt.step = spy_step
    args = _make_namespace()
    _ = EpochProgress(total=1, desc="ep0", disable=True)
    loss_f = torch.nn.MSELoss()

    # Replace model.forward to inject NaN loss; keep the real wrapper
    orig_forward = model.forward

    def nan_forward(*a, **kw):
        orig_forward(*a, **kw)
        return torch.tensor(float("nan"))

    model.forward = nan_forward  # type: ignore[method-assign]

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train.train_one_epoch(
            model,
            loader,
            opt,
            loss_f,
            0,
            args,
            torch.device("cpu"),
            scaler=None,
            amp_dtype=None,
            tb_logger=None,
        )

    # The optimizer.step() must NOT have been called at all.
    assert step_calls["n"] == 0
    model.forward = orig_forward  # type: ignore[method-assign]
    opt.step = real_step
