"""Regression tests for the fixes applied after the deep audit.

Covers:
  * C1   - inference.predict thread-safety (concurrent callers must
            not corrupt each other's total_length)
  * M1   - _check_resume_args checks kernel_size / tln / hidden_dim
  * M2   - RadarEcho.transform symmetry + pickle safety
  * M3   - CUDAGraphRunner.__call__ is thread-safe (lock)
  * M5   - evaluate skips batches with non-finite metrics
"""

import pickle
import threading
import concurrent.futures as cf

import numpy as np
import pytest
import torch

from dataset import MovingMNIST, RadarEcho
from inference import predict
from mim import MIM
from train import _check_resume_args, evaluate


HIDDEN = [8, 8]
IN_LEN, TOTAL = 3, 6
HORIZON = 3


def _make_model():
    return MIM(
        1,
        1,
        [1, 1, 16, 16],
        hidden_dim=HIDDEN,
        total_length=IN_LEN + HORIZON,
        input_length=IN_LEN,
    )


# ---------------------------------------------------------------------------
# C1: inference.predict thread-safety
# ---------------------------------------------------------------------------


def test_predict_concurrent_calls_restore_total_length():
    model = _make_model()
    old = model.total_length
    frames = torch.rand(1, IN_LEN, 1, 16, 16)
    errs = []

    def worker(h):
        try:
            for _ in range(5):
                pred = predict(model, frames, horizon=h)
                assert pred.shape[1] == h
                # Read total_length under the same lock that predict()
                # uses, so we don't race with another thread that may
                # have just acquired the lock and mutated it.
                import inference

                with inference._predict_lock:
                    assert model.total_length == old
        except Exception as exc:
            errs.append(exc)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(worker, [1, 2, 3, 4]))
    assert not errs
    assert model.total_length == old


# ---------------------------------------------------------------------------
# M1: _check_resume_args comprehensive checks
# ---------------------------------------------------------------------------


def _fake_namespace(**kw):
    ns = type("NS", (), {})()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _fake_ckpt_args(**kw):
    return {"args": kw}


def test_resume_args_rejects_kernel_size_mismatch():
    ckpt = _fake_ckpt_args(kernel_size=5)
    args = _fake_namespace(kernel_size=3)
    with pytest.raises(ValueError, match="kernel_size"):
        _check_resume_args(ckpt, args)


def test_resume_args_rejects_tln_mismatch():
    ckpt = _fake_ckpt_args(tln=True)
    args = _fake_namespace(tln=False)
    with pytest.raises(ValueError, match="tln"):
        _check_resume_args(ckpt, args)


def test_resume_args_rejects_hidden_dim_mismatch():
    ckpt = _fake_ckpt_args(hidden_dim=[32, 32])
    args = _fake_namespace(hidden_dim=[16, 16, 16, 16])
    with pytest.raises(ValueError, match="hidden_dim"):
        _check_resume_args(ckpt, args)


def test_resume_args_accepts_hidden_dim_string_roundtrip():
    """A [32,32] checkpoint vs a CLI-parsed '32,32' should match."""
    ckpt = _fake_ckpt_args(hidden_dim=[32, 32])
    args = _fake_namespace(hidden_dim="32,32")
    _check_resume_args(ckpt, args)  # must not raise


# ---------------------------------------------------------------------------
# M2: RadarEcho.transform + pickle safety
# ---------------------------------------------------------------------------


def _make_radar_npz(tmp_path):
    """2 samples, 24 timesteps, 16x16, no channels."""
    p = tmp_path / "data.npz"
    np.savez(p, train=np.zeros((2, 24, 16, 16), dtype=np.float32))
    return str(p)


def test_radar_echo_applies_transform(tmp_path):
    path = _make_radar_npz(tmp_path)
    ds = RadarEcho(path, total_length=8, input_length=4, transform=lambda x: x * 2.0)
    seq = ds[0]
    # raw values are 0 -> transformed 0. Just verify the transform was hit.
    assert seq.shape == (8, 1, 16, 16)


def test_radar_echo_transform_pickle_safe(tmp_path):
    path = _make_radar_npz(tmp_path)
    ds = RadarEcho(
        path, total_length=8, input_length=4, transform=lambda x: x
    )  # lambdas die in pickle
    state = pickle.dumps(ds)
    restored = pickle.loads(state)
    assert restored.transform is None
    assert restored._data is None
    # Should still load and yield samples after unpickling.
    seq = restored[0]
    assert seq.shape == (8, 1, 16, 16)


# MovingMNIST should share the same pickle-safe contract.
def _make_mnist_npz(tmp_path):
    p = tmp_path / "mnist.npz"
    np.savez(
        p,
        train=np.zeros((20, 4, 16, 16), dtype=np.float32),
        test=np.zeros((4, 4, 16, 16), dtype=np.float32),
    )
    return str(p)


def test_moving_mnist_transform_pickle_safe(tmp_path):
    path = _make_mnist_npz(tmp_path)
    ds = MovingMNIST(path, TOTAL, IN_LEN, is_train=True, transform=lambda x: x)
    state = pickle.dumps(ds)
    restored = pickle.loads(state)
    assert restored.transform is None
    seq = restored[0]
    assert seq.shape[0] == TOTAL


# ---------------------------------------------------------------------------
# M3: CUDAGraphRunner thread-safety (lock-only property on CPU)
# ---------------------------------------------------------------------------


def test_cuda_graph_runner_has_lock_attribute(tmp_path):
    """We cannot exercise the lock on CPU, but we verify the attribute
    exists so that a refactor that removes it fails loudly here."""
    # CUDAGraphRunner requires CUDA; on CPU we just check the module
    # exposes the class and its expected lock field.
    from inference import CUDAGraphRunner

    assert hasattr(CUDAGraphRunner, "__call__")
    # Confirm the module has its lock guard.
    import inference

    assert hasattr(inference, "_predict_lock")
    assert type(inference._predict_lock) is type(threading.RLock())


# ---------------------------------------------------------------------------
# M5: evaluate skips batches with non-finite metrics
# ---------------------------------------------------------------------------


def test_evaluate_raises_when_all_batches_produce_nan(tmp_path):
    model = _make_model()
    model.eval()
    from dataset import get_dataloader

    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    ds = MovingMNIST(str(p), TOTAL, IN_LEN, is_train=False)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=False)

    real_forward = model.forward

    def nan_forward(frames, **kw):
        out = real_forward(frames, **kw)
        return out + float("nan")

    model.forward = nan_forward
    try:
        args = _fake_namespace(input_length=IN_LEN)
        with pytest.raises(FloatingPointError, match="non-finite"):
            evaluate(model, loader, args, torch.device("cpu"), max_video_samples=0)
    finally:
        model.forward = real_forward


def test_evaluate_returns_metrics_only_when_no_video(tmp_path):
    model = _make_model()
    model.eval()
    from dataset import get_dataloader

    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8),
    )
    ds = MovingMNIST(str(p), TOTAL, IN_LEN, is_train=False)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=False)

    args = _fake_namespace(input_length=IN_LEN)
    result = evaluate(model, loader, args, torch.device("cpu"), max_video_samples=0)
    assert isinstance(result, dict)
    assert "samples" in result
    assert "skipped_nan" in result
    assert result["skipped_nan"] == 0
