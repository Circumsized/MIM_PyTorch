import os
import threading

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _load_split(data_path, split):
    """Load a sequence array from ``.npy`` or ``.npz``.

    ``.npy`` files are memory-mapped: workers share OS page-cache pages
    instead of each holding a private copy. ``.npz`` splits are read fully
    (compressed archives cannot be mmapped).
    """
    data_path = str(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"dataset file not found: {data_path}")
    if data_path.lower().endswith(".npy"):
        return np.load(data_path, mmap_mode="r")
    if not data_path.lower().endswith(".npz"):
        raise ValueError(
            f"unsupported dataset format: {data_path} (expected .npy or .npz)"
        )
    with np.load(data_path) as f:
        if split not in f.files:
            raise KeyError(
                f"key '{split}' not found in {data_path}; available keys: {f.files}"
            )
        arr = f[split]
        # .npz cannot be mmapped: every DataLoader worker holds a full
        # decompressed copy. Surface the footprint so operators can size
        # num_workers accordingly (prefer uncompressed .npy for large sets).
        gb = arr.nbytes / 1e9
        if gb >= 0.1:
            print(
                f"Note: .npz split {split!r} is {gb:.2f} GB decompressed; "
                f"each DataLoader worker keeps a full copy in RAM"
            )
        return arr


def _normalize_to_unit(seq):
    """Normalize a numpy sequence to float32 in [0, 1].

    ``uint8`` data is divided by 255. Float data is expected to already be
    in [0, 1]; out-of-range values raise so a mis-scaled dataset cannot
    silently corrupt training and downstream metrics (which clamp to
    ``data_range``).
    """
    if seq.dtype == np.uint8:
        seq = seq.astype(np.float32)
        seq /= 255.0
        return seq
    seq = seq.astype(np.float32, copy=False)
    if seq.size and not np.isfinite(seq).all():
        raise ValueError(
            "float data contains NaN/Inf; finite values are required so a "
            "corrupted sample cannot silently poison training and metrics"
        )
    if seq.size and (seq.min() < 0.0 or seq.max() > 1.0):
        raise ValueError(
            f"float data must be pre-normalized to [0, 1], got range "
            f"[{float(seq.min()):.4g}, {float(seq.max()):.4g}]; normalize "
            f"the dataset or store it as uint8"
        )
    return seq


class _LazySeqDataset(Dataset):
    """Base class that avoids pickling the data array into every worker.

    On spawn-based start methods (Windows) the DataLoader pickles the
    dataset object into each worker process. Keeping ``_data`` out of
    ``__getstate__`` means only the file path travels; each worker then
    re-loads the array lazily on first access (with ``persistent_workers``
    this happens once per training run, not per epoch). On fork-based
    start methods (Linux) the already-loaded array is inherited via COW.
    """

    def __getstate__(self):
        # Never pickle _data (large array) or _data_lock (threading.Lock is
        # not pickle-able). Also exclude ``transform`` — it is commonly a
        # lambda or a `torchvision.transforms.Compose` wrapping lambdas, and
        # neither survives pickling across DataLoader workers. Each worker
        # re-attaches its own transform via ``worker_init_fn``.
        state = self.__dict__.copy()
        state["_data"] = None
        state.pop("_data_lock", None)
        state.pop("transform", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Attributes intentionally dropped from the pickled state get
        # back to their defaults. ``transform`` must be None because the
        # real value (a lambda or Compose) cannot travel across pickling.
        self._data = None
        self._data_lock = threading.Lock()
        if not hasattr(self, "transform"):
            self.transform = None

    def _ensure_data(self):
        if self._data is None:
            lock = getattr(self, "_data_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._data_lock = lock
            with lock:
                if self._data is None:
                    self._data = _load_split(self.data_path, self.split)
        return self._data


class MovingMNIST(_LazySeqDataset):
    """Moving MNIST, layout [T, N, H, W] (official npz: keys train/test)."""

    def __init__(
        self, data_path, total_length=20, input_length=10, is_train=True, transform=None
    ):
        self.data_path = str(data_path)
        self.split = "train" if is_train else "test"
        self.total_length = total_length
        self.input_length = input_length
        self.transform = transform
        self._data = None
        self._data_lock = threading.Lock()

        if total_length < 1:
            raise ValueError(f"total_length must be >= 1, got {total_length}")

        data = self._ensure_data()
        if data.ndim != 4:
            raise ValueError(f"expected 4-D [T, N, H, W] data, got shape {data.shape}")
        if data.shape[0] < total_length:
            raise ValueError(
                f"sequence length {data.shape[0]} < total_length={total_length}"
            )
        self.num_samples = int(data.shape[1])
        if self.num_samples < 1:
            raise ValueError(f"dataset has no samples (shape {data.shape})")
        self.frame_shape = (1, int(data.shape[2]), int(data.shape[3]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples")
        data = self._ensure_data()
        seq = np.ascontiguousarray(data[: self.total_length, idx])
        seq = torch.from_numpy(_normalize_to_unit(seq)).unsqueeze(1)
        if self.transform is not None:
            seq = self.transform(seq)
        return seq


class RadarEcho(_LazySeqDataset):
    """Radar echo sequences, layout [N, T, H, W] or [N, T, C, H, W]."""

    def __init__(
        self, data_path, total_length=20, input_length=10, is_train=True, transform=None
    ):
        self.data_path = str(data_path)
        self.split = "train" if is_train else "test"
        self.total_length = total_length
        self.input_length = input_length
        self.transform = transform
        self._data = None
        self._data_lock = threading.Lock()

        if total_length < 1:
            raise ValueError(f"total_length must be >= 1, got {total_length}")

        data = self._ensure_data()
        if data.ndim not in (4, 5):
            raise ValueError(
                f"expected [N, T, H, W] or [N, T, C, H, W], got {data.shape}"
            )
        if data.shape[1] < total_length:
            raise ValueError(
                f"sequence length {data.shape[1]} < total_length={total_length}"
            )
        self.num_samples = int(data.shape[0])
        if self.num_samples < 1:
            raise ValueError(f"dataset has no samples (shape {data.shape})")
        self.frame_shape = (
            (1, int(data.shape[2]), int(data.shape[3]))
            if data.ndim == 4
            else tuple(int(v) for v in data.shape[2:])
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples")
        data = self._ensure_data()
        seq = data[idx][: self.total_length]
        if not seq.flags.writeable:
            seq = seq.copy()
        seq = torch.from_numpy(_normalize_to_unit(seq))
        if seq.dim() == 3:
            seq = seq.unsqueeze(1)
        if self.transform is not None:
            seq = self.transform(seq)
        return seq


def recommend_num_workers(has_gpu=None):
    """Heuristic DataLoader worker count that avoids CPU oversubscription.

    With GPU training the main process is mostly idle, so workers can use
    half the cores (capped at 8). CPU-only training must share cores with
    the main process' intra-op thread pool, so fewer workers are used.
    """
    cpu = os.cpu_count() or 1
    if has_gpu is None:
        has_gpu = torch.cuda.is_available()
    if has_gpu:
        return max(1, min(8, cpu // 2))
    return max(1, min(4, cpu // 4))


def _worker_init_fn(worker_id):
    # One intra-op thread per worker: N workers x 1 thread leaves the
    # remaining cores to the main process instead of N x default pools.
    torch.set_num_threads(1)
    # DataLoader re-seeds torch RNG per worker but NOT numpy.
    np.random.seed((torch.initial_seed() + worker_id) % (2**32))


def get_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=0,
    prefetch_factor=2,
    drop_last=True,
    pin_memory=None,
    worker_init_fn=_worker_init_fn,
):
    """Build a DataLoader with worker/thread friendly defaults.

    ``pin_memory=None`` auto-enables when CUDA is available. Pass an explicit
    ``False`` for CPU-only training to skip the pinned-memory copy thread.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(dataset, **kwargs)
