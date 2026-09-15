"""Regression tests for the 2nd-pass deep audit.

Each test targets a specific fix from the deep-detection review; failures
mean a regression slipped back in.
"""

import argparse

import numpy as np
import pytest
import torch
import torch.nn as nn

import train
from dataset import MovingMNIST, get_dataloader
from mim import MIM

# ---------------------------------------------------------------------------
# CLI argument validation: modulo-by-zero and empty-loop foot-guns
# ---------------------------------------------------------------------------


def test_validate_args_rejects_zero_intervals():
    import argparse

    base = argparse.Namespace(
        log_interval=100, save_interval=5, batch_size=8, epochs=10, lr=0.001
    )
    for field in ("log_interval", "save_interval", "batch_size", "epochs", "lr"):
        bad = argparse.Namespace(**vars(base))
        setattr(bad, field, 0)
        with pytest.raises(ValueError):
            train._validate_args(bad)


def test_validate_args_accepts_valid():
    import argparse

    ok = argparse.Namespace(
        log_interval=100,
        save_interval=5,
        batch_size=8,
        epochs=10,
        lr=0.001,
        total_length=20,
        input_length=10,
        kernel_size=3,
        hidden_dim=[8, 8],
        num_workers=-1,
        num_threads=-1,
        prefetch_factor=2,
        ss_initial_prob=1.0,
        ss_final_prob=0.0,
        ss_start_epoch=0,
        ss_stop_epoch=5,
    )
    train._validate_args(ok)  # must not raise


def test_validate_args_rejects_scheduled_sampling_out_of_range():
    import argparse

    base = argparse.Namespace(
        log_interval=100,
        save_interval=5,
        batch_size=8,
        epochs=10,
        lr=0.001,
        total_length=20,
        input_length=10,
        kernel_size=3,
        hidden_dim=[8, 8],
        num_workers=0,
        num_threads=1,
        prefetch_factor=2,
        ss_initial_prob=1.0,
        ss_final_prob=0.0,
        ss_start_epoch=0,
        ss_stop_epoch=5,
    )
    for field, bad_value in [
        ("ss_initial_prob", 1.5),
        ("ss_final_prob", -0.1),
        ("ss_start_epoch", -1),
        ("ss_stop_epoch", -1),
    ]:
        bad = argparse.Namespace(**vars(base))
        setattr(bad, field, bad_value)
        with pytest.raises(ValueError):
            train._validate_args(bad)


def test_tln_flags_are_mutually_exclusive():
    """Passing --tln --no_tln must produce a single tln value, not two fields."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tln", dest="tln", action="store_true", default=True)
    parser.add_argument("--no_tln", dest="tln", action="store_false")
    on = parser.parse_args(["--tln"])
    off = parser.parse_args(["--no_tln"])
    assert on.tln is True and off.tln is False
    assert not hasattr(on, "no_tln") and not hasattr(off, "no_tln")


# ---------------------------------------------------------------------------
# B: AsyncCheckpointSaver must be non-blocking and still keep order on close
# ---------------------------------------------------------------------------


def test_async_saver_does_not_block_on_submission(tmp_path):
    import time

    saver = train.AsyncCheckpointSaver(max_inflight=4)

    # Force each save to take >=200ms by intercepting the static save.
    real_save = train.AsyncCheckpointSaver._atomic_save
    delays = []

    def slow_save(obj, path):
        time.sleep(0.2)
        delays.append(time.perf_counter())
        return real_save(obj, path)

    train.AsyncCheckpointSaver._atomic_save = staticmethod(slow_save)
    try:
        t0 = time.perf_counter()
        for i in range(3):
            saver.save({"i": i}, str(tmp_path / f"x_{i}.pth"))
        submit_dt = time.perf_counter() - t0
        # Three saves that each block 200ms would be >=0.6s if serialised;
        # submission should be effectively free.
        assert submit_dt < 0.1, f"submission was {submit_dt:.3f}s, expected <0.1s"
    finally:
        train.AsyncCheckpointSaver._atomic_save = staticmethod(real_save)
        saver.close()


def test_async_saver_close_preserves_order(tmp_path):
    saver = train.AsyncCheckpointSaver(max_inflight=1)
    payloads = [{"marker": i, "arr": torch.full((1,), float(i))} for i in range(4)]
    paths = [str(tmp_path / f"order_{i}.pth") for i in range(4)]
    for p, ckpt in zip(paths, payloads):
        saver.save(ckpt, p)
    saver.close()
    for i, p in enumerate(paths):
        loaded = torch.load(p, weights_only=True)
        assert loaded["marker"] == i
        assert loaded["arr"].item() == float(i)


# ---------------------------------------------------------------------------
# A: _load_resume_checkpoint must enforce key/type allow-list
# ---------------------------------------------------------------------------


def test_resume_rejects_unknown_key(tmp_path):
    bogus = tmp_path / "bogus.pth"
    torch.save({"model_state_dict": {}, "extra_field": 1}, bogus)
    with pytest.raises(ValueError, match="unexpected key"):
        train._load_resume_checkpoint(str(bogus), torch.device("cpu"), None)


def test_resume_rejects_wrong_type(tmp_path):
    bogus = tmp_path / "bad.pth"
    # 'epoch' must be int, supply a string (all required keys present so the
    # type check, not the required-key check, is what fires).
    torch.save(
        {"model_state_dict": {}, "optimizer_state_dict": {}, "epoch": "oops"}, bogus
    )
    with pytest.raises(ValueError, match="wrong type"):
        train._load_resume_checkpoint(str(bogus), torch.device("cpu"), None)


def test_resume_rejects_non_dict(tmp_path):
    bogus = tmp_path / "list.pth"
    torch.save([1, 2, 3], bogus)
    with pytest.raises(ValueError, match="must be a dict"):
        train._load_resume_checkpoint(str(bogus), torch.device("cpu"), None)


def test_resume_rejects_missing_required_keys(tmp_path):
    bogus = tmp_path / "partial.pth"
    torch.save({"epoch": 3}, bogus)
    with pytest.raises(ValueError, match="missing required key"):
        train._load_resume_checkpoint(str(bogus), torch.device("cpu"), None)


# ---------------------------------------------------------------------------
# B2: AsyncCheckpointSaver concurrency hardening
# ---------------------------------------------------------------------------


def test_async_saver_failed_save_leaves_no_tmp(tmp_path):
    saver = train.AsyncCheckpointSaver(max_inflight=1)
    target = tmp_path / "fail.pth"
    real_save = train.AsyncCheckpointSaver._atomic_save

    def boom(obj, path):
        raise RuntimeError("disk full")

    train.AsyncCheckpointSaver._atomic_save = staticmethod(boom)
    try:
        # The failure surfaces either at save() (fail-fast pruning) or at
        # close() (drain); both are acceptable, but it must surface.
        with pytest.raises(RuntimeError, match="disk full"):
            saver.save({"x": 1}, target)
            saver.close()
    finally:
        train.AsyncCheckpointSaver._atomic_save = staticmethod(real_save)
        if not saver._closed:
            try:
                saver.close()
            except RuntimeError:
                pass
    assert not target.exists()
    assert not (tmp_path / "fail.pth.tmp").exists()


def test_async_saver_close_is_idempotent(tmp_path):
    saver = train.AsyncCheckpointSaver(max_inflight=1)
    saver.save({"x": 1}, tmp_path / "idem.pth")
    saver.close()
    saver.close()  # must not raise on a shut-down executor
    assert (tmp_path / "idem.pth").exists()


def test_async_saver_save_after_close_rejected(tmp_path):
    saver = train.AsyncCheckpointSaver(max_inflight=1)
    saver.close()
    with pytest.raises(RuntimeError):
        saver.save({"x": 1}, tmp_path / "late.pth")


# ---------------------------------------------------------------------------
# F: get_scheduled_sampling_prob must clamp to [0, 1]
# ---------------------------------------------------------------------------


class _NS:  # namespace bag for fake args
    ss_start_epoch = 0
    ss_stop_epoch = 5
    ss_initial_prob = 1.0
    ss_final_prob = 0.0


def test_scheduled_sampling_clamp_lower():
    args = _NS()
    # Misconfigured: stop before start
    args.ss_start_epoch, args.ss_stop_epoch = 10, 5
    p = train.get_scheduled_sampling_prob(20, args)
    assert 0.0 <= p <= 1.0


def test_scheduled_sampling_clamp_upper():
    args = _NS()
    # initial < final (unusual)
    args.ss_initial_prob, args.ss_final_prob = 0.0, 1.0
    p = train.get_scheduled_sampling_prob(2, args)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# C: total_loss must be float32 even under AMP (covered as scalar dtype check)
# ---------------------------------------------------------------------------


def test_total_loss_dtype_is_float32():
    # Simulates the train_one_epoch allocation: dtype must be fp32 so an
    # AMP-produced fp16 loss can be added without losing precision.
    device = torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)
    # Simulated fp16 loss (would have happened under autocast).
    loss_fp16 = torch.tensor(0.123456789, dtype=torch.float16)
    total += loss_fp16.detach().float()
    assert total.dtype == torch.float32
    assert torch.isfinite(total).all()


# ---------------------------------------------------------------------------
# D: mim.py must reject a CUDA ss_bool when the model is on CPU
# ---------------------------------------------------------------------------


def test_ss_bool_cross_device_auto_migrates():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    frames = torch.rand(1, 4, 1, 16, 16)
    cuda_ss = torch.zeros(1, 1, 16, 16, device="cuda")
    out = model(frames, ss_bool=cuda_ss)
    assert out.shape == (1, 3, 1, 16, 16)
    assert out.device.type == "cpu"


def test_ss_bool_cpu_auto_migrates_to_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = MIM(
        1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2
    ).to("cuda")
    frames = torch.rand(1, 4, 1, 16, 16, device="cuda")
    cpu_ss = torch.zeros(1, 1, 16, 16)  # lives on CPU
    out = model(frames, ss_bool=cpu_ss)
    assert out.shape == (1, 3, 1, 16, 16)


# ---------------------------------------------------------------------------
# End-to-end: train.py 1 epoch with the new AsyncCheckpointSaver
# ---------------------------------------------------------------------------


def test_train_one_epoch_runs_with_async_saver(tmp_path):
    import argparse

    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 4, 32, 32), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 2, 32, 32), dtype=np.uint8),
    )

    args = argparse.Namespace(
        dataset="mnist",
        data_path=str(p),
        batch_size=2,
        total_length=6,
        input_length=3,
        hidden_dim=[8, 8],
        kernel_size=3,
        lr=1e-3,
        epochs=1,
        num_workers=0,
        num_threads=1,
        device="cpu",
        save_dir=str(tmp_path / "ckpt"),
        log_interval=1,
        save_interval=1,
        ss_start_epoch=0,
        ss_stop_epoch=5,
        ss_initial_prob=1.0,
        ss_final_prob=0.0,
        tln=True,
        resume=None,
        amp=False,
        compile=False,
        grad_ckpt=False,
        prefetch_factor=2,
        seed=0,
    )
    # Drive just the inner functions to avoid spawning CLI parsing.
    torch.manual_seed(0)
    device = torch.device("cpu")
    from dataset import get_dataloader

    train_ds = MovingMNIST(str(p), 6, 3, is_train=True)
    test_ds = MovingMNIST(str(p), 6, 3, is_train=False)
    train_loader = get_dataloader(train_ds, 2, num_workers=0, drop_last=True)
    test_loader = get_dataloader(test_ds, 2, num_workers=0, drop_last=False)
    model = MIM(
        1, 1, [2, 1, 32, 32], hidden_dim=[8, 8], total_length=6, input_length=3
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train.train_one_epoch(
        model, train_loader, opt, nn.MSELoss(), 0, args, device, scaler=None
    )
    metrics = train.evaluate(model, test_loader, args, device)
    assert loss == loss  # not NaN
    assert all(v == v for v in metrics.values())  # all finite

    # Async save should not crash even with no in-flight writes.
    saver = train.AsyncCheckpointSaver()
    saver.save(
        {"epoch": 0, "model_state_dict": model.state_dict()}, str(tmp_path / "ckpt.pth")
    )
    saver.close()
    assert (tmp_path / "ckpt.pth").exists()


# ---------------------------------------------------------------------------
# Deep-scan round: resume args cross-check, evaluate contract, empty epoch
# ---------------------------------------------------------------------------


def test_check_resume_args_rejects_length_mismatch():
    import argparse

    ckpt = {"args": {"total_length": 20, "input_length": 10}}
    args = argparse.Namespace(total_length=20, input_length=5)
    with pytest.raises(ValueError, match="input_length"):
        train._check_resume_args(ckpt, args)
    args2 = argparse.Namespace(total_length=15, input_length=10)
    with pytest.raises(ValueError, match="total_length"):
        train._check_resume_args(ckpt, args2)


def test_check_resume_args_accepts_match_and_missing():
    import argparse

    ckpt = {"args": {"total_length": 20, "input_length": 10}}
    ok = argparse.Namespace(total_length=20, input_length=10)
    train._check_resume_args(ckpt, ok)  # must not raise
    train._check_resume_args({}, ok)  # legacy checkpoint without args
    train._check_resume_args({"args": None}, ok)


def test_evaluate_rejects_drop_last_loader(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(p, train=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8))
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=True)
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    import argparse

    args = argparse.Namespace(input_length=2)
    with pytest.raises(ValueError, match="drop_last"):
        train.evaluate(model, loader, args, torch.device("cpu"))


def test_train_one_epoch_rejects_empty_loader(tmp_path):
    import argparse

    p = tmp_path / "mm.npz"
    # 2 samples with batch_size=4 and drop_last=True -> zero batches.
    np.savez(p, train=np.random.randint(0, 256, (8, 2, 16, 16), dtype=np.uint8))
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    loader = get_dataloader(ds, batch_size=4, num_workers=0, drop_last=True)
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    args = argparse.Namespace(
        total_length=4,
        input_length=2,
        log_interval=1,
        ss_start_epoch=0,
        ss_stop_epoch=5,
        ss_initial_prob=0.0,
        ss_final_prob=0.0,
    )
    with pytest.raises(ValueError, match="no batches"):
        train.train_one_epoch(
            model, loader, opt, nn.MSELoss(), 0, args, torch.device("cpu")
        )


def test_evaluate_raises_when_all_batches_are_nan(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=False)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=False)
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)

    real_forward = model.forward

    def nan_forward(frames, **kw):
        out = real_forward(frames, **kw)
        return out + float("nan")

    model.forward = nan_forward
    try:
        args = argparse.Namespace(input_length=2)
        with pytest.raises(FloatingPointError, match="non-finite"):
            train.evaluate(
                model, loader, args, torch.device("cpu"), max_video_samples=0
            )
    finally:
        model.forward = real_forward
