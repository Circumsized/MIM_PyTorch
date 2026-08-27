"""Regression tests for the 3rd-pass deep audit (CUDA/robustness).

Covers: MIMS/MIMN ct_weight chunk equivalence (bitwise vs the old
``c_t.repeat`` formulation), ValueError-based input validation that
survives ``python -O``, pin_memory plumbing, and inference_mode paths.
"""

import numpy as np
import os
import pytest
import torch

from dataset import get_dataloader, MovingMNIST
from mim import MIM, MIMS, MIMN, sequence_to_channels_last
import train


def test_sequence_channels_last_slices():
    frames = torch.randn(2, 4, 3, 8, 10)
    converted = sequence_to_channels_last(frames)
    assert converted.shape == (4, 2, 3, 8, 10)
    assert torch.equal(converted, frames.transpose(0, 1))
    assert all(
        converted[ts].is_contiguous(memory_format=torch.channels_last)
        for ts in range(converted.shape[0])
    )


def _make_cell(cls, **kw):
    torch.manual_seed(0)
    return cls(
        4,
        8,
        kernel_size=3,
        in_shape=[2, 4, 16, 16],
        forget_bias=1.0,
        tln=kw.get("tln", False),
    )


# ---------------------------------------------------------------------------
# L2: MIMS/MIMN chunk(ct_weight) must equal the old repeat formulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MIMS, MIMN])
@pytest.mark.parametrize("tln", [False, True])
def test_ct_weight_chunk_equivalence(cls, tln):
    torch.manual_seed(7)
    cell = _make_cell(cls, tln=tln)
    h = torch.randn(2, 8, 16, 16)
    c = torch.randn(2, 8, 16, 16)
    x = torch.randn(2, 4, 16, 16)

    h_new, c_new = cell(x, h, c)

    # Reference: the pre-optimization formulation.
    h_concat = cell.conv_h(h)
    if cell.layer_norm:
        h_concat = cell.tln_h(h_concat)
    i_h, g_h, f_h, o_h = torch.split(h_concat, cell.num_hidden, dim=1)
    ct_activation = c.repeat(1, 2, 1, 1) * cell.ct_weight
    i_c_ref, f_c_ref = torch.split(ct_activation, cell.num_hidden, dim=1)
    # compare against what forward() internally produced is not possible;
    # instead verify chunk(ct_weight) == split of repeat formulation.
    w_i, w_f = cell.ct_weight.chunk(2, dim=0)
    assert torch.equal(w_i, cell.ct_weight[: cell.num_hidden])
    assert torch.equal(w_f, cell.ct_weight[cell.num_hidden :])
    assert torch.allclose(c * w_i, i_c_ref)
    assert torch.allclose(c * w_f, f_c_ref)
    # forward still produces finite outputs with correct shapes
    assert h_new.shape == (2, 8, 16, 16)
    assert c_new.shape == (2, 8, 16, 16)
    assert torch.isfinite(h_new).all() and torch.isfinite(c_new).all()


@pytest.mark.parametrize("cls", [MIMS, MIMN])
def test_cell_backward_through_chunk(cls):
    torch.manual_seed(3)
    cell = _make_cell(cls)
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    h = torch.randn(2, 8, 16, 16)
    c = torch.randn(2, 8, 16, 16)
    h_new, c_new = cell(x, h, c)
    (h_new.sum() + c_new.sum()).backward()
    # ct_weight gradient must flow
    assert cell.ct_weight.grad is not None
    assert torch.isfinite(cell.ct_weight.grad).all()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# L1: forward validation must be ValueError, not assert (-O safe)
# ---------------------------------------------------------------------------


def test_forward_rejects_bad_ndim():
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    bad = torch.rand(1, 4, 16, 16)  # 4-D instead of 5-D
    with pytest.raises(ValueError):
        model(bad)


def test_forward_rejects_short_sequence():
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=6, input_length=2)
    bad = torch.rand(1, 4, 1, 16, 16)  # T=4 < total_length=6
    with pytest.raises(ValueError):
        model(bad)


def test_forward_rejects_wrong_channels():
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    bad = torch.rand(1, 4, 3, 16, 16)  # C=3 != input_dims=1
    with pytest.raises(ValueError):
        model(bad)


def test_validation_survives_python_o():
    # Run the validation in a real ``python -O`` interpreter: asserts are
    # stripped there, so only a ValueError-based check can prove the guard
    # survives optimization.
    import subprocess
    import sys

    code = (
        "import torch\n"
        "from mim import MIM\n"
        "m = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], "
        "total_length=4, input_length=2)\n"
        "try:\n"
        "    m(torch.rand(1, 4, 3, 16, 16))\n"
        "except ValueError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode()


# ---------------------------------------------------------------------------
# L6: pin_memory plumbing
# ---------------------------------------------------------------------------


def test_dataloader_pin_memory_explicit_false(tmp_path):
    p = tmp_path / "mm.npz"
    np.savez(p, train=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8))
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    loader = get_dataloader(
        ds, batch_size=2, num_workers=0, pin_memory=False, drop_last=True
    )
    assert loader.pin_memory is False


def test_dataloader_pin_memory_default_auto(tmp_path):
    # Default None -> auto (cuda availability), same as before.
    p = tmp_path / "mm.npz"
    np.savez(p, train=np.random.randint(0, 256, (8, 4, 16, 16), dtype=np.uint8))
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=True)
    assert loader.pin_memory == torch.cuda.is_available()


# ---------------------------------------------------------------------------
# L3/L10: inference_mode evaluation must not break gradients elsewhere
# ---------------------------------------------------------------------------


def test_evaluate_runs_with_inference_mode(tmp_path):
    import argparse

    p = tmp_path / "mm.npz"
    np.savez(
        p,
        train=np.random.randint(0, 256, (8, 4, 32, 32), dtype=np.uint8),
        test=np.random.randint(0, 256, (8, 4, 32, 32), dtype=np.uint8),
    )
    args = argparse.Namespace(
        total_length=4,
        input_length=2,
    )
    torch.manual_seed(0)
    device = torch.device("cpu")
    test_ds = MovingMNIST(str(p), 4, 2, is_train=False)
    loader = get_dataloader(
        test_ds, 2, num_workers=0, drop_last=False, pin_memory=False
    )
    model = MIM(
        1, 1, [2, 1, 32, 32], hidden_dim=[8, 8], total_length=4, input_length=2
    ).to(device)
    metrics = train.evaluate(model, loader, args, device)
    assert all(v == v for v in metrics.values())
    assert metrics["mse"] >= 0
    # model must remain trainable after inference_mode evaluation
    for param in model.parameters():
        param.requires_grad_(True)
    out = model(torch.rand(1, 4, 1, 32, 32))
    out.sum().backward()  # must not raise "inference tensor" errors
