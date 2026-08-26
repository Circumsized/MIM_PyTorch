import torch
import pytest
from mim import MIM


def test_malicious_checkpoint_blocked(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("__import__('os').system('echo pwned')",))
    p = tmp_path / "evil.pth"
    torch.save({'model_state_dict': Evil()}, p)
    with pytest.raises(Exception):
        torch.load(p, weights_only=True)


def test_input_validation():
    model = MIM(1, 1, [2, 1, 32, 32], hidden_dim=[8, 8],
                total_length=6, input_length=3)
    bad = torch.randn(2, 6, 3, 32, 32)
    with pytest.raises(AssertionError):
        model(bad)
