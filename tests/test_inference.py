import numpy as np
import pytest
import torch

from inference import (
    CUDAGraphRunner,
    infer_architecture,
    load_input,
    load_model,
    pad_frames,
    predict,
)
from mim import MIM

HIDDEN = [8, 8]
IN_LEN, TOTAL = 3, 6


def make_model_and_ckpt(tmp_path, horizon=3):
    model = MIM(
        1,
        1,
        [1, 1, 32, 32],
        hidden_dim=HIDDEN,
        total_length=IN_LEN + horizon,
        input_length=IN_LEN,
    )
    ckpt_path = tmp_path / "mim.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return model, str(ckpt_path)


def test_infer_architecture(tmp_path):
    model, ckpt = make_model_and_ckpt(tmp_path)
    sd = model.state_dict()
    input_dims, hidden, k, h, w, tln = infer_architecture(sd)
    assert input_dims == 1
    assert hidden == HIDDEN
    assert k == 3
    assert (h, w) == (32, 32)
    assert tln is True


def test_load_model_roundtrip(tmp_path):
    _, ckpt = make_model_and_ckpt(tmp_path)
    model = load_model(ckpt, device="cpu", total_length=TOTAL, input_length=IN_LEN)
    assert model.input_length == IN_LEN
    assert model.hidden_dim == HIDDEN
    assert isinstance(model, MIM)


def test_pad_frames():
    frames = torch.randn(2, 4, 1, 32, 32)
    padded = pad_frames(frames, 8)
    assert padded.shape == (2, 8, 1, 32, 32)
    # Padding repeats the last observed frame.
    assert torch.equal(padded[:, 4], frames[:, 3])
    assert pad_frames(frames, 4) is frames


def test_predict_output_shape_and_finite(tmp_path):
    _, ckpt = make_model_and_ckpt(tmp_path, horizon=3)
    model = load_model(ckpt, device="cpu", total_length=TOTAL, input_length=IN_LEN)
    frames = torch.rand(2, IN_LEN, 1, 32, 32)
    pred = predict(model, frames, horizon=3)
    assert pred.shape == (2, 3, 1, 32, 32)
    assert torch.isfinite(pred).all()


def test_predict_restores_total_length(tmp_path):
    _, ckpt = make_model_and_ckpt(tmp_path, horizon=3)
    model = load_model(ckpt, device="cpu", total_length=TOTAL, input_length=IN_LEN)
    old = model.total_length
    predict(model, torch.rand(1, IN_LEN, 1, 32, 32), horizon=1)
    assert model.total_length == old


def test_predict_input_validation(tmp_path):
    _, ckpt = make_model_and_ckpt(tmp_path)
    model = load_model(ckpt, device="cpu", total_length=TOTAL, input_length=IN_LEN)
    with pytest.raises(ValueError):
        predict(model, torch.rand(1, 2, 1, 32, 32), horizon=1)  # too short
    with pytest.raises(ValueError):
        predict(model, torch.rand(1, IN_LEN, 1, 32, 32), horizon=0)


def test_load_input_shapes(tmp_path):
    p = tmp_path / "in.npy"
    np.save(p, np.random.rand(12, 32, 32).astype(np.float32))
    f = load_input(str(p), input_length=4)
    assert f.shape == (1, 12, 1, 32, 32)

    p2 = tmp_path / "in_c.npy"
    np.save(p2, np.random.rand(12, 2, 32, 32).astype(np.float32))
    f2 = load_input(str(p2), input_length=4)
    assert f2.shape == (1, 12, 2, 32, 32)

    p3 = tmp_path / "in_b.npy"
    np.save(p3, np.random.rand(3, 12, 1, 32, 32).astype(np.float32))
    f3 = load_input(str(p3), input_length=4)
    assert f3.shape == (3, 12, 1, 32, 32)


def test_load_input_uint8_normalization(tmp_path):
    p = tmp_path / "in_u8.npy"
    np.save(p, np.random.randint(0, 256, (8, 32, 32)).astype(np.uint8))
    f = load_input(str(p), input_length=4)
    assert f.dtype == torch.float32
    assert f.max() <= 1.0


def test_load_input_too_short(tmp_path):
    p = tmp_path / "short.npy"
    np.save(p, np.random.rand(2, 32, 32).astype(np.float32))
    with pytest.raises(ValueError):
        load_input(str(p), input_length=4)


def test_load_input_rejects_integer_dtype(tmp_path):
    p = tmp_path / "int.npy"
    np.save(p, np.random.randint(0, 256, (8, 32, 32)).astype(np.int32))
    with pytest.raises(ValueError, match="unsupported input dtype"):
        load_input(str(p), input_length=4)


def test_load_input_rejects_bad_ndim(tmp_path):
    p = tmp_path / "scalar.npy"
    np.save(p, np.random.rand(32, 32).astype(np.float32))  # 2-D
    with pytest.raises(ValueError, match="ndim"):
        load_input(str(p), input_length=4)


def test_infer_architecture_rejects_non_mim_dict():
    with pytest.raises(ValueError, match="does not look like a MIM"):
        infer_architecture({"unrelated.weight": torch.zeros(1)})


def test_infer_architecture_single_layer_needs_explicit_hw():
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[8], total_length=4, input_length=2)
    sd = model.state_dict()
    with pytest.raises(ValueError, match="--height/--width"):
        infer_architecture(sd)
    input_dims, hidden, k, h, w, tln = infer_architecture(sd, height=16, width=16)
    assert (hidden, h, w) == ([8], 16, 16)


def test_infer_architecture_rejects_bad_explicit_hw():
    model = MIM(1, 1, [1, 1, 16, 16], hidden_dim=[8], total_length=4, input_length=2)
    with pytest.raises(ValueError, match="height/width"):
        infer_architecture(model.state_dict(), height=0, width=16)


def test_load_model_rejects_input_length_mismatch(tmp_path):
    model = MIM(
        1, 1, [1, 1, 32, 32], hidden_dim=HIDDEN, total_length=TOTAL, input_length=IN_LEN
    )
    ckpt_path = tmp_path / "mim_args.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {"input_length": IN_LEN, "total_length": TOTAL},
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="input_length"):
        load_model(
            str(ckpt_path), device="cpu", total_length=TOTAL, input_length=IN_LEN + 1
        )


def test_cuda_graph_runner_thread_safety(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = MIM(
        1, 1, [1, 1, 16, 16], hidden_dim=[4, 4], total_length=4, input_length=2
    ).cuda()
    example = torch.rand(1, 4, 1, 16, 16, device="cuda")
    runner = CUDAGraphRunner(model, example)

    errs = []

    def worker(seed):
        try:
            torch.manual_seed(seed)
            frames = torch.rand(1, 4, 1, 16, 16, device="cuda")
            out1 = runner(frames)
            out2 = runner(frames)
            assert out1.shape == (1, 2, 1, 16, 16)
            assert torch.allclose(out1, out2, atol=1e-6)
        except Exception as exc:
            errs.append(exc)

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(worker, range(8)))
    assert not errs
