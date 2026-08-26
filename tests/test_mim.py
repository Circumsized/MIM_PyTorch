import torch

from mim import MIM

INPUT_DIMS = 1
OUT_DIMS = 1
IN_SHAPE = [2, 1, 32, 32]
HIDDEN_DIM = [8, 8]
TOTAL_LENGTH = 6
INPUT_LENGTH = 3


def make_model():
    return MIM(INPUT_DIMS, OUT_DIMS, IN_SHAPE, hidden_dim=HIDDEN_DIM,
               total_length=TOTAL_LENGTH, input_length=INPUT_LENGTH)


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
    ss_bool = torch.randint(
        0, 2, (2, TOTAL_LENGTH - INPUT_LENGTH - 1, 32, 32)).float()
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
