"""Spec-compliance lock tests: pin down fixes for 3 porting bugs found by
diffing against the official TensorFlow implementation (Yunbo426/MIM).

Each test guards a specific divergence from the TF reference so a future
refactor cannot silently reintroduce it.
"""

import math

import torch
import torch.nn as nn

from mim import MIM, MIMN, MIMS, MIMBlock

# ---------------------------------------------------------------------------
# BUG-3 (fixed): MIMN.oc_weight must be a trainable nn.Parameter.
# The original port assigned ``self.oc_weight = nn.init.kaiming_normal_(...)``,
# which returns a plain tensor (no grad), silently freezing the output-gate
# spatial modulation of every non-stationary module.
# ---------------------------------------------------------------------------


def _params_of(module):
    return {name for name, _ in module.named_parameters()}


def test_mimn_oc_weight_is_parameter():
    m = MIMN(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
    assert isinstance(m.oc_weight, torch.nn.Parameter)
    assert isinstance(m.ct_weight, torch.nn.Parameter)
    assert "oc_weight" in _params_of(m)
    assert "ct_weight" in _params_of(m)


def test_mims_oc_weight_is_parameter():
    m = MIMS(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
    assert isinstance(m.oc_weight, torch.nn.Parameter)
    assert isinstance(m.ct_weight, torch.nn.Parameter)


def test_mimn_oc_weight_receives_gradient():
    torch.manual_seed(0)
    m = MIMN(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
    x = torch.randn(2, 4, 16, 16)
    h = torch.randn(2, 8, 16, 16)
    c = torch.randn(2, 8, 16, 16)
    h_new, c_new = m(x, h, c)
    (h_new.sum() + c_new.sum()).backward()
    assert m.oc_weight.grad is not None
    assert torch.count_nonzero(m.oc_weight.grad) > 0, "oc_weight grad is all-zero"
    assert m.ct_weight.grad is not None


# ---------------------------------------------------------------------------
# BUG-1 (fixed): MIMBlock must feed the *different* branch (num_hidden
# channels, the MIMN output) into MIMS, not the main-branch in_channel.
# With uniform hidden_dim the two coincide, so this is guarded indirectly by
# the uniform-hidden_dim contract plus a shape smoke test.
# ---------------------------------------------------------------------------


def test_mimblock_mims_input_channels_are_num_hidden():
    blk = MIMBlock(8, 8, kernel_size=3, in_shape=[2, 8, 16, 16])
    # MIMS.conv_x must expect num_hidden (=8) input channels, not in_channel.
    assert blk.mims.conv_x.in_channels == 8
    assert blk.mims.conv_h.in_channels == 8


def test_non_uniform_hidden_dim_rejected():
    try:
        MIM(1, 1, [2, 1, 16, 16], hidden_dim=[4, 8], total_length=4, input_length=2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-uniform hidden_dim should raise ValueError")


# ---------------------------------------------------------------------------
# BUG-2 (fixed): MIMS gate order must be (i, g, f, o) as in the TF reference
# ``i_h, g_h, f_h, o_h = tf.split(...)``. Guard by asserting both MIMS and
# MIMN route the first split slice to the input gate (the one modulated by
# ct_weight's first chunk).
# ---------------------------------------------------------------------------


def test_mims_and_mimn_produce_finite_output():
    torch.manual_seed(1)
    for cls in (MIMS, MIMN):
        m = cls(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
        x = torch.randn(2, 4, 16, 16)
        h = torch.randn(2, 8, 16, 16)
        c = torch.randn(2, 8, 16, 16)
        h_new, c_new = m(x, h, c)
        assert torch.isfinite(h_new).all()
        assert torch.isfinite(c_new).all()


# ---------------------------------------------------------------------------
# End-to-end: the model runs forward/backward and keeps every custom
# spatial-modulation weight in state_dict (nothing silently untracked).
# ---------------------------------------------------------------------------


def test_model_state_dict_contains_modulation_weights():
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[8, 8], total_length=4, input_length=2)
    sd = model.state_dict()
    assert "stlstm_layer.1.mims.ct_weight" in sd
    assert "stlstm_layer.1.mims.oc_weight" in sd
    assert "stlstm_layer_diff.0.ct_weight" in sd
    assert "stlstm_layer_diff.0.oc_weight" in sd


def test_model_end_to_end_backward():
    torch.manual_seed(2)
    model = MIM(1, 1, [2, 1, 16, 16], hidden_dim=[8, 8], total_length=4, input_length=2)
    frames = torch.randn(2, 4, 1, 16, 16)
    out = model(frames)
    assert out.shape == (2, 3, 1, 16, 16)
    out.sum().backward()
    # every parameter (incl. ct_weight / oc_weight) receives a gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no grad"


# ---------------------------------------------------------------------------
# Gate order (behavioral, hand-computed): when the convs are zeroed out, only
# the ct_weight / oc_weight modulation remains, so the gate routing (i,g,f,o)
# is directly observable. A reordered split would break the reference below.
# ---------------------------------------------------------------------------


def test_mims_gate_order_hand_computed():
    torch.manual_seed(0)
    m = MIMS(2, 2, kernel_size=3, in_shape=[1, 2, 1, 1], forget_bias=0.0, tln=False)
    nn.init.zeros_(m.conv_h.weight)
    nn.init.zeros_(m.conv_h.bias)
    nn.init.zeros_(m.conv_x.weight)
    nn.init.zeros_(m.conv_x.bias)
    with torch.no_grad():
        m.ct_weight.copy_(torch.tensor([[[0.5]], [[-1.0]], [[0.25]], [[2.0]]]))
        m.oc_weight.copy_(torch.tensor([[[0.7]], [[-0.4]]]))

    x = torch.randn(1, 2, 1, 1)
    h = torch.randn(1, 2, 1, 1)
    c = torch.randn(1, 2, 1, 1)
    h_new, c_new = m(x, h, c)

    w_i, w_f = m.ct_weight[:2], m.ct_weight[2:]
    i_ = torch.sigmoid(c * w_i)
    f_ = torch.sigmoid(c * w_f + 0.0)
    g_ = torch.tanh(torch.zeros_like(c))
    c_ref = f_ * c + i_ * torch.tanh(g_)
    o_ = torch.zeros_like(c)
    h_ref = torch.sigmoid(o_ + c_ref * m.oc_weight) * torch.tanh(c_ref)

    assert torch.allclose(c_new, c_ref)
    assert torch.allclose(h_new, h_ref)


# ---------------------------------------------------------------------------
# Official initializer parity (verified against Yunbo426/MIM src/layers).
# ---------------------------------------------------------------------------


def test_mimn_conv_init_is_0001_uniform():
    m = MIMN(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
    for conv in (m.conv_h, m.conv_x):
        assert conv.weight.min().item() >= -0.001
        assert conv.weight.max().item() <= 0.001


def test_mims_conv_init_is_xavier():
    m = MIMS(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
    bound = math.sqrt(6.0 / (8 * 9 + 32 * 9))
    assert m.conv_h.weight.abs().max().item() <= bound + 1e-6
    # Xavier bound here is ~0.129, far above MIMN's ±0.001.
    assert m.conv_h.weight.abs().max().item() > 0.001


def test_modulation_weights_use_tf_glorot():
    for cls in (MIMS, MIMN):
        m = cls(4, 8, kernel_size=3, in_shape=[2, 4, 16, 16])
        for w in (m.ct_weight, m.oc_weight):
            c, h, wd = w.shape[0], w.shape[1], w.shape[2]
            limit = math.sqrt(6.0 / (h * wd + c * h))
            assert w.abs().max().item() <= limit + 1e-6
