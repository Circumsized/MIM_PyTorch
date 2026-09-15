"""Adversarial depth-audit (round 2) regression + property-based tests.

Each test targets a specific adversarial surface (A1–A11) that was probed
empirically, reasoned about, and — where a real defect existed — fixed.
The property-based cases use Hypothesis to assert invariants that hold for
arbitrary valid inputs, while the deterministic cases pin the exact
adversarial input that must now be rejected.
"""

import threading

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

import inference
import train
from dataset import _normalize_to_unit
from inference import infer_architecture

# ---------------------------------------------------------------------------
# A1: _normalize_to_unit must reject non-finite float data (was silently
#     passed through, poising training since NaN/Inf comparisons are False).
# ---------------------------------------------------------------------------


def test_normalize_to_unit_rejects_nan():
    arr = np.array([[np.nan, 1.0], [0.5, np.inf]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN/Inf"):
        _normalize_to_unit(arr)


def test_normalize_to_unit_rejects_inf():
    arr = np.array([1.0, np.inf, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN/Inf"):
        _normalize_to_unit(arr)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
@settings(deadline=None, max_examples=50)
def test_normalize_to_unit_accepts_finite_in_range(vals):
    arr = np.array(vals, dtype=np.float32)
    out = _normalize_to_unit(arr)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------------------------------------------------------------------------
# A5: generate_ss_bool must reject a negative scheduled-sampling length
#     (was an opaque "negative dimension" RuntimeError from torch.rand).
# ---------------------------------------------------------------------------


def test_generate_ss_bool_rejects_negative_ss_length():
    with pytest.raises(ValueError, match="ss_length"):
        train.generate_ss_bool(
            2,
            total_length=10,
            input_length=10,
            height=16,
            width=16,
            prob=0.5,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# A6: AsyncCheckpointSaver._snapshot must not serialize an nn.Module by
#     reference (would silently snapshot a live graph mutated by optimizer).
# ---------------------------------------------------------------------------


def test_snapshot_rejects_module():
    with pytest.raises(TypeError, match=r"nn\.Module"):
        train.AsyncCheckpointSaver._snapshot({"mod": nn.Linear(2, 2)})


# ---------------------------------------------------------------------------
# A8: _predict_lock must be re-entrant (RLock) so a same-thread nested call
#     (e.g. a transform that calls predict on the same model) cannot deadlock.
# ---------------------------------------------------------------------------


def test_predict_lock_is_reentrant():
    # threading.RLock is a factory, not a type; compare against the actual
    # type of a freshly-created RLock. A plain Lock would deadlock here.
    assert type(inference._predict_lock) is type(threading.RLock())
    with inference._predict_lock, inference._predict_lock:
        pass  # must not deadlock; would on a plain threading.Lock


# ---------------------------------------------------------------------------
# A11: infer_architecture must reject a malformed x_cc whose output channels
#     are not a multiple of 4 (was silent integer truncation via // 4).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_ch", [1, 2, 3, 5, 6, 7, 9])
def test_infer_architecture_rejects_non_multiple_of_4(out_ch):
    sd = {"stlstm_layer.0.x_cc.weight": torch.zeros(out_ch, 1, 3, 3)}
    with pytest.raises(ValueError, match="multiple of 4"):
        infer_architecture(sd, height=16, width=16)


def test_infer_architecture_accepts_multiple_of_4():
    sd = {
        "stlstm_layer.0.x_cc.weight": torch.zeros(8, 1, 3, 3),
        "stlstm_layer.0.tln_t.gamma": torch.ones(1, 32, 1, 1),
    }
    input_dims, hidden, k, h, w, tln = infer_architecture(sd, height=16, width=16)
    assert hidden == [2]
    assert input_dims == 1
    assert tln is True


# ---------------------------------------------------------------------------
# V6: infer_architecture must reject malformed state_dict values
#     (non-tensor / non-4D / non-square kernel) with a clear ValueError.
# ---------------------------------------------------------------------------


def test_infer_architecture_rejects_non_tensor_value():
    sd = {"stlstm_layer.0.x_cc.weight": "bad"}
    with pytest.raises(ValueError, match="4-D conv weight"):
        infer_architecture(sd, height=16, width=16)


def test_infer_architecture_rejects_non_4d_weight():
    sd = {"stlstm_layer.0.x_cc.weight": torch.zeros(8, 3)}
    with pytest.raises(ValueError, match="4-D conv weight"):
        infer_architecture(sd, height=16, width=16)


def test_infer_architecture_rejects_non_square_kernel():
    sd = {"stlstm_layer.0.x_cc.weight": torch.zeros(8, 1, 3, 5)}
    with pytest.raises(ValueError, match="square kernels"):
        infer_architecture(sd, height=16, width=16)


# ---------------------------------------------------------------------------
# Cross-cutting property invariants (valid inputs must not throw).
# ---------------------------------------------------------------------------


@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=1, max_value=8),
    h=st.integers(min_value=1, max_value=16),
    w=st.integers(min_value=1, max_value=16),
)
@settings(deadline=None, max_examples=30)
def test_mim_forward_shape_invariant(batch, hidden, h, w):
    from mim import MIM

    total, input_length = 4, 2
    model = MIM(
        1,
        1,
        [batch, 1, h, w],
        hidden_dim=[hidden, hidden],
        total_length=total,
        input_length=input_length,
    )
    frames = torch.rand(batch, total, 1, h, w)
    out = model(frames)
    assert out.shape == (batch, total - 1, 1, h, w)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# D2: predict must bound horizon (unbounded horizon pads/rolls to OOM).
# ---------------------------------------------------------------------------


def test_predict_rejects_excessive_horizon():
    import inference
    from mim import MIM

    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    frames = torch.rand(1, 2, 1, 16, 16)
    with pytest.raises(ValueError, match="safe limit"):
        inference.predict(model, frames, horizon=inference._MAX_HORIZON + 1)


# ---------------------------------------------------------------------------
# D3: checkpoint loading must bound file size (zip-bomb-style oversized
#     tensor deserialization DoS).
# ---------------------------------------------------------------------------


def test_load_model_rejects_oversized_checkpoint(tmp_path, monkeypatch):
    import inference

    monkeypatch.setattr(inference, "_MAX_CKPT_BYTES", 10)
    p = tmp_path / "big.pth"
    p.write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="safe-load limit"):
        inference.load_model(str(p), device="cpu")


def test_load_resume_checkpoint_rejects_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_MAX_CKPT_BYTES", 10)
    p = tmp_path / "big.pth"
    p.write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="safe-load limit"):
        train._load_resume_checkpoint(str(p), torch.device("cpu"), None)


# ---------------------------------------------------------------------------
# R1: sequence_to_channels_last must be value-correct even on degenerate
#     shapes (C==1 / H==W==1), not only stride-correct.
# ---------------------------------------------------------------------------


@given(
    batch=st.integers(min_value=1, max_value=4),
    total=st.integers(min_value=2, max_value=5),
    channels=st.sampled_from([1, 2, 4]),
    h=st.integers(min_value=1, max_value=8),
    w=st.integers(min_value=1, max_value=8),
)
@settings(deadline=None, max_examples=40)
def test_sequence_to_channels_last_value_invariant(batch, total, channels, h, w):
    from mim import sequence_to_channels_last

    frames = torch.rand(batch, total, channels, h, w)
    out = sequence_to_channels_last(frames)
    ref = frames.transpose(0, 1)
    for ts in range(total):
        assert torch.equal(
            out[ts].contiguous(memory_format=torch.contiguous_format), ref[ts]
        )


# ---------------------------------------------------------------------------
# R2: eager forward outputs must not alias across successive calls.
# ---------------------------------------------------------------------------


def test_eager_forward_outputs_do_not_alias():
    from mim import MIM

    model = MIM(
        1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2
    ).eval()
    frames = torch.rand(1, 4, 1, 16, 16)
    with torch.no_grad():
        o1 = model(frames)
        saved = o1.clone()
        model(frames)
    assert torch.equal(o1, saved)


# ---------------------------------------------------------------------------
# R3: predict() must reject a torch.compile-wrapped model (attribute
#     mutation shadowing leads to stale-graph behaviour).
# ---------------------------------------------------------------------------


def test_predict_rejects_torch_compiled_model():
    from mim import MIM

    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2)
    compiled = torch.compile(model, backend="eager")
    with pytest.raises(ValueError, match=r"torch\.compile"):
        inference.predict(compiled, torch.rand(1, 2, 1, 16, 16), horizon=1)


# ---------------------------------------------------------------------------
# R4: gradient NaN guard must block optimizer.step() in the non-scaler path
#     (bf16 / plain fp32 training lacked GradScaler's NaN-skip).
# ---------------------------------------------------------------------------


def test_check_grad_norm_finite_rejects_nan():
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        train._check_grad_norm_finite(torch.tensor(float("nan")), epoch=0, batch_idx=0)


def test_check_grad_norm_finite_accepts_finite():
    train._check_grad_norm_finite(torch.tensor(1.5), epoch=0, batch_idx=0)


def test_train_one_epoch_aborts_on_nan_gradients(tmp_path):
    import argparse

    from dataset import MovingMNIST, get_dataloader
    from mim import MIM

    p = tmp_path / "mm.npz"
    np.savez(p, train=np.random.randint(0, 256, (8, 6, 16, 16), dtype=np.uint8))
    ds = MovingMNIST(str(p), total_length=6, input_length=3, is_train=True)
    loader = get_dataloader(ds, batch_size=2, num_workers=0, drop_last=True)
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[4, 4], total_length=6, input_length=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    real_forward = model.forward

    def nan_grad_hook(grad):
        # Keeps the forward loss finite (so the pre-backward loss check
        # passes) while forcing the gradient total-norm to NaN.
        return torch.full_like(grad, float("nan"))

    first_param = next(model.parameters())
    hook = first_param.register_hook(nan_grad_hook)
    args = argparse.Namespace(
        total_length=6,
        input_length=3,
        log_interval=1,
        ss_start_epoch=0,
        ss_stop_epoch=5,
        ss_initial_prob=0.0,
        ss_final_prob=0.0,
    )
    try:
        with pytest.raises(FloatingPointError, match="gradient norm"):
            train.train_one_epoch(
                model, loader, opt, nn.MSELoss(), 0, args, torch.device("cpu")
            )
    finally:
        hook.remove()
        model.forward = real_forward
    assert all(torch.isfinite(p).all() for p in model.parameters())
