import os
import pickle

import numpy as np
import pytest
import torch

from dataset import (
    MovingMNIST,
    RadarEcho,
    _load_split,
    get_dataloader,
    recommend_num_workers,
)


def make_fake_npz(path, n=8, length=12, h=32, w=32):
    data = np.random.randint(0, 256, (length, n, h, w), dtype=np.uint8)
    np.savez_compressed(path, train=data, test=data)
    return data


def make_fake_npy(path, n=8, length=12, h=32, w=32):
    data = np.random.randint(0, 256, (length, n, h, w), dtype=np.uint8)
    np.save(path, data)
    return data


def test_load_split_npy_mmap(tmp_path):
    p = tmp_path / "train.npy"
    make_fake_npy(p)
    arr = _load_split(str(p), "train")
    assert isinstance(arr, np.memmap)


def test_load_split_npz(tmp_path):
    p = tmp_path / "data.npz"
    make_fake_npz(p)
    arr = _load_split(str(p), "train")
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 4


def test_load_split_missing_key(tmp_path):
    p = tmp_path / "data.npz"
    np.savez(p, train=np.zeros((4, 2, 8, 8), dtype=np.uint8))
    with pytest.raises(KeyError):
        _load_split(str(p), "test")


def test_load_split_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_split(str(tmp_path / "nope.npz"), "train")


def test_moving_mnist_sample(tmp_path):
    p = tmp_path / "mnist.npz"
    make_fake_npz(p, n=6, length=10)
    ds = MovingMNIST(str(p), total_length=8, input_length=4, is_train=True)
    assert len(ds) == 6
    assert ds.frame_shape == (1, 32, 32)
    sample = ds[0]
    assert sample.shape == (8, 1, 32, 32)
    assert sample.dtype == torch.float32
    assert 0.0 <= sample.min() and sample.max() <= 1.0


def test_moving_mnist_rejects_short_sequences(tmp_path):
    p = tmp_path / "mnist.npz"
    make_fake_npz(p, length=4)
    with pytest.raises(ValueError):
        MovingMNIST(str(p), total_length=8, input_length=4)


def test_lazy_pickle_keeps_data_out(tmp_path):
    """Workers must not inherit the data array via pickle (Windows spawn)."""
    p = tmp_path / "mnist.npz"
    make_fake_npz(p, n=4, length=6)
    ds = MovingMNIST(str(p), total_length=6, input_length=3)
    assert ds._data is not None  # loaded eagerly in the main process

    state = pickle.loads(pickle.dumps(ds))
    assert state._data is None  # only the path travels to the worker
    sample = state[0]  # lazy reload on first access
    assert sample.shape == (6, 1, 32, 32)


def test_radar_echo_sample_4d(tmp_path):
    p = tmp_path / "radar.npy"
    data = np.random.rand(5, 10, 16, 16).astype(np.float32)
    np.save(p, data)
    ds = RadarEcho(str(p), total_length=6, input_length=3, is_train=True)
    assert len(ds) == 5
    assert ds.frame_shape == (1, 16, 16)
    assert ds[0].shape == (6, 1, 16, 16)


def test_radar_echo_sample_5d(tmp_path):
    p = tmp_path / "radar5.npy"
    data = np.random.rand(5, 10, 2, 16, 16).astype(np.float32)
    np.save(p, data)
    ds = RadarEcho(str(p), total_length=6, input_length=3, is_train=True)
    assert ds.frame_shape == (2, 16, 16)
    assert ds[0].shape == (6, 2, 16, 16)


def test_recommend_num_workers_bounds():
    cpu = os.cpu_count() or 1
    assert recommend_num_workers(True) == max(1, min(8, cpu // 2))
    assert recommend_num_workers(False) == max(1, min(4, cpu // 4))


def test_dataloader_batching(tmp_path):
    p = tmp_path / "mnist.npz"
    make_fake_npz(p, n=8, length=6)
    ds = MovingMNIST(str(p), total_length=6, input_length=3)
    loader = get_dataloader(
        ds, batch_size=4, shuffle=False, num_workers=0, drop_last=True
    )
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0].shape == (4, 6, 1, 32, 32)


def test_load_split_unsupported_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("not a dataset")
    with pytest.raises(ValueError, match="unsupported dataset format"):
        _load_split(str(p), "train")


def test_moving_mnist_zero_total_length_rejected(tmp_path):
    p = tmp_path / "mm.npz"
    make_fake_npz(p, n=2, length=4)
    with pytest.raises(ValueError, match="total_length"):
        MovingMNIST(str(p), total_length=0, input_length=1)


def test_moving_mnist_out_of_range_index(tmp_path):
    p = tmp_path / "mm.npz"
    make_fake_npz(p, n=2, length=4)
    ds = MovingMNIST(str(p), total_length=4, input_length=2)
    with pytest.raises(IndexError):
        ds[99]


def test_radar_echo_out_of_range_index(tmp_path):
    p = tmp_path / "radar.npy"
    data = np.random.rand(3, 4, 1, 8, 8).astype(np.float32)
    np.save(p, data)
    ds = RadarEcho(str(p), total_length=4, input_length=2)
    with pytest.raises(IndexError):
        ds[99]


def test_get_dataloader_rejects_bad_params():
    ds = object()  # the validator runs before the loader touches the dataset
    with pytest.raises(ValueError, match="batch_size"):
        get_dataloader(ds, batch_size=0, num_workers=0)
    with pytest.raises(ValueError, match="num_workers"):
        get_dataloader(ds, batch_size=1, num_workers=-1)
    with pytest.raises(ValueError, match="prefetch_factor"):
        get_dataloader(ds, batch_size=1, num_workers=0, prefetch_factor=0)


def test_moving_mnist_float_data_in_unit_range_accepted(tmp_path):
    p = tmp_path / "mm.npy"
    data = np.random.rand(6, 4, 16, 16).astype(np.float32)  # already [0,1]
    np.save(p, data)
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    sample = ds[0]
    assert sample.dtype == torch.float32
    assert 0.0 <= sample.min() and sample.max() <= 1.0


def test_moving_mnist_float_data_out_of_range_rejected(tmp_path):
    p = tmp_path / "mm.npy"
    # Float data in [0, 255]: not pre-normalized. Dividing blindly by 255
    # would hide the mistake; the loader must reject it instead.
    data = np.random.rand(6, 4, 16, 16).astype(np.float32) * 255.0
    np.save(p, data)
    ds = MovingMNIST(str(p), total_length=4, input_length=2, is_train=True)
    with pytest.raises(ValueError, match="pre-normalized"):
        ds[0]


def test_radar_echo_uint8_normalized_to_unit(tmp_path):
    p = tmp_path / "radar.npy"
    data = np.random.randint(0, 256, (4, 6, 16, 16)).astype(np.uint8)
    np.save(p, data)
    ds = RadarEcho(str(p), total_length=4, input_length=2, is_train=True)
    sample = ds[0]
    assert sample.dtype == torch.float32
    assert 0.0 <= sample.min() and sample.max() <= 1.0


def test_radar_echo_float_out_of_range_rejected(tmp_path):
    p = tmp_path / "radar.npy"
    data = (np.random.rand(4, 6, 16, 16) * 100.0).astype(np.float32)
    np.save(p, data)
    ds = RadarEcho(str(p), total_length=4, input_length=2, is_train=True)
    with pytest.raises(ValueError, match="pre-normalized"):
        ds[0]
