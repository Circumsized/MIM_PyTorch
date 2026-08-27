import pytest
import torch

from mim import MIM, MIMBlock

INPUT_DIMS = 1
OUT_DIMS = 1
IN_SHAPE = [2, 1, 32, 32]
HIDDEN_DIM = [8, 8]
TOTAL_LENGTH = 6
INPUT_LENGTH = 3


def make_model():
    return MIM(
        INPUT_DIMS,
        OUT_DIMS,
        IN_SHAPE,
        hidden_dim=HIDDEN_DIM,
        total_length=TOTAL_LENGTH,
        input_length=INPUT_LENGTH,
    )


def make_frames(batch_size):
    return torch.rand(batch_size, TOTAL_LENGTH, INPUT_DIMS, 32, 32)


def test_output_shape():
    model = make_model()
    frames = make_frames(2)
    out = model(frames)
    assert out.shape == (2, TOTAL_LENGTH - 1, OUT_DIMS, 32, 32)


def test_ss_bool_none_path():
    model = make_model()
    frames = make_frames(2)
    out = model(frames, ss_bool=None)
    assert out.shape == (2, TOTAL_LENGTH - 1, OUT_DIMS, 32, 32)
    assert torch.isfinite(out).all()


def test_ss_bool_provided_path():
    model = make_model()
    frames = make_frames(2)
    ss_bool = torch.randint(0, 2, (2, TOTAL_LENGTH - INPUT_LENGTH - 1, 32, 32)).float()
    out = model(frames, ss_bool=ss_bool)
    assert out.shape == (2, TOTAL_LENGTH - 1, OUT_DIMS, 32, 32)
    assert torch.isfinite(out).all()


def test_dynamic_batch():
    model = make_model()
    model.eval()
    out2 = model(make_frames(2))
    out4 = model(make_frames(4))
    assert out2.shape[0] == 2
    assert out4.shape[0] == 4
    assert out2.shape[1:] == out4.shape[1:]


def test_gradient_flow():
    model = make_model()
    frames = make_frames(2)
    out = model(frames)
    loss = out.mean()
    loss.backward()
    params = list(model.parameters())
    assert len(params) > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_state_dict_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "mim.pth"
    torch.save(model.state_dict(), path)

    model2 = make_model()
    model2.load_state_dict(torch.load(path, weights_only=True))

    model.eval()
    model2.eval()
    frames = make_frames(2)
    with torch.no_grad():
        out1 = model(frames)
        out2 = model2(frames)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_oc_ct_weight_in_named_parameters():
    model = make_model()
    names = [name for name, _ in model.named_parameters()]
    assert any(name.endswith("oc_weight") for name in names)
    assert any(name.endswith("ct_weight") for name in names)


def _backward_once(gradient_checkpointing):
    torch.manual_seed(0)
    model = make_model()
    model.gradient_checkpointing = gradient_checkpointing
    frames = make_frames(2)
    out = model(frames)
    out.pow(2).mean().backward()
    return model


def test_gradient_checkpoint_equivalence():
    """Checkpointed recomputation must produce identical gradients."""
    plain = _backward_once(False)
    ckpt = _backward_once(True)
    for (n1, p1), (_, p2) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert p1.grad is not None and p2.grad is not None, n1
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
            f"gradient mismatch for {n1}"
        )


def test_gradient_checkpoint_eval_noop():
    """In eval mode checkpointing must be skipped (no grad, no recompute)."""
    model = make_model()
    model.gradient_checkpointing = True
    model.eval()
    with torch.no_grad():
        out = model(make_frames(2))
    assert out.shape == (2, TOTAL_LENGTH - 1, OUT_DIMS, 32, 32)


def test_non_uniform_hidden_dim_rejected():
    with pytest.raises(ValueError):
        MIM(1, 1, [2, 1, 32, 32], hidden_dim=[8, 16], total_length=6, input_length=3)


def test_bad_lengths_rejected():
    with pytest.raises(ValueError):
        MIM(1, 1, [2, 1, 32, 32], hidden_dim=[8, 8], total_length=3, input_length=3)


def test_even_kernel_size_rejected():
    with pytest.raises(ValueError):
        MIM(
            1,
            1,
            [2, 1, 32, 32],
            hidden_dim=[8, 8],
            total_length=6,
            input_length=3,
            kernel_size=4,
        )


def test_non_unit_stride_rejected():
    with pytest.raises(ValueError):
        MIM(
            1,
            1,
            [2, 1, 32, 32],
            hidden_dim=[8, 8],
            total_length=6,
            input_length=3,
            stride=2,
        )


def test_zero_input_length_rejected():
    with pytest.raises(ValueError):
        MIM(1, 1, [2, 1, 32, 32], hidden_dim=[8, 8], total_length=6, input_length=0)


def test_mim_rejects_zero_or_negative_channels():
    with pytest.raises(ValueError, match="input_dims"):
        MIM(0, 1, [2, 1, 32, 32], hidden_dim=[4, 4], total_length=4, input_length=2)
    with pytest.raises(ValueError, match="out_dims"):
        MIM(1, 0, [2, 1, 32, 32], hidden_dim=[4, 4], total_length=4, input_length=2)


def test_mim_rejects_empty_or_bad_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim"):
        MIM(1, 1, [2, 1, 32, 32], hidden_dim=[], total_length=4, input_length=2)
    with pytest.raises(ValueError, match="hidden_dim"):
        MIM(1, 1, [2, 1, 32, 32], hidden_dim=[0, 0], total_length=4, input_length=2)


def test_mim_rejects_bad_in_shape():
    with pytest.raises(ValueError, match="in_shape"):
        MIM(1, 1, [2, 1, 0, 32], hidden_dim=[4, 4], total_length=4, input_length=2)
    with pytest.raises(ValueError, match="in_shape"):
        MIM(1, 1, [2, 1, 32], hidden_dim=[4, 4], total_length=4, input_length=2)


def test_mim_rejects_non_numeric_forget_bias():
    with pytest.raises(ValueError, match="forget_bias"):
        MIM(
            1,
            1,
            [2, 1, 32, 32],
            hidden_dim=[4, 4],
            total_length=4,
            input_length=2,
            forget_bias="one",
        )


def test_bad_ss_bool_shape_rejected():
    model = make_model()
    frames = make_frames(2)
    bad = torch.zeros(2, 99, 32, 32)
    with pytest.raises(ValueError):
        model(frames, ss_bool=bad)


def test_init_states_layout_and_count():
    for hidden_dim, expected in [([8, 8], 8), ([8, 8, 8], 13)]:
        model = MIM(
            1,
            1,
            [2, 1, 32, 32],
            hidden_dim=hidden_dim,
            total_length=TOTAL_LENGTH,
            input_length=3,
        )
        assert model._num_states == expected
        states = model._init_states(2, torch.device("cpu"))
        assert len(states) == expected
        assert all(s.shape == (2, hidden_dim[0], 32, 32) for s in states)
        assert all(s.dtype == torch.float32 for s in states)
        assert all(torch.count_nonzero(s) == 0 for s in states)


def test_fp16_forward_is_consistent():
    """States/ss_bool must follow the model dtype, else model.half() would
    crash with a conv dtype mismatch. Regresses the fp16 inference bug."""
    model = MIM(
        1, 1, [2, 1, 8, 8], hidden_dim=[4, 4], total_length=4, input_length=2
    ).half()
    states = model._init_states(2, torch.device("cpu"))
    assert all(s.dtype == torch.float16 for s in states)
    frames = torch.randn(2, 4, 1, 8, 8).half()
    out = model(frames, ss_bool=None)
    assert out.dtype == torch.float16
    assert out.shape == (2, 3, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_forward_rejects_spatial_mismatch():
    model = make_model()  # built for 32x32
    bad = torch.rand(2, TOTAL_LENGTH, INPUT_DIMS, 16, 16)
    with pytest.raises(ValueError, match="spatial size mismatch"):
        model(bad)


def test_ss_zero_masks_nan_frames():
    """With ss=0 the masked frames must never be read: under IEEE semantics
    ``0 * NaN == NaN``, so the old ``ss*frames + (1-ss)*x_gen`` formulation
    would poison x_gen. torch.where must keep the output finite."""
    model = make_model()
    model.eval()
    frames = make_frames(2)
    frames[:, INPUT_LENGTH:] = float("nan")  # tail frames are all masked
    ss_bool = torch.zeros(2, TOTAL_LENGTH - INPUT_LENGTH - 1, 32, 32)
    with torch.no_grad():
        out = model(frames, ss_bool=ss_bool)
    assert torch.isfinite(out).all()


def test_mimblock_last_respects_bias_flag():
    block = MIMBlock(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16], bias=False)
    assert block.last.bias is None
    block_b = MIMBlock(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16], bias=True)
    assert block_b.last.bias is not None
