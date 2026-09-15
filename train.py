import argparse
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from dataset import MovingMNIST, RadarEcho, get_dataloader, recommend_num_workers
from metrics import batch_mae, batch_mse, batch_psnr, batch_ssim
from mim import MIM
from visualization import EpochProgress, TBLogger


class AsyncCheckpointSaver:
    """Serialize checkpoints off the training thread.

    The device->CPU snapshot happens synchronously (cheap memcpy), while
    the potentially slow disk write runs on a background thread. Snapshots
    are deep copies, so subsequent optimizer steps can never corrupt an
    in-flight save. ``save`` is fully non-blocking: at most one snapshot is
    prepared at a time, but up to ``max_inflight`` writes may be in flight
    simultaneously; ``close()`` drains them in submission order.
    """

    def __init__(self, max_inflight=2):
        self._executor = ThreadPoolExecutor(max_workers=max_inflight)
        self._pending = []  # FIFO of Futures in submission order.
        self._closed = False

    @staticmethod
    def _snapshot(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, nn.Module):
            # A bare nn.Module in a checkpoint would be serialized by
            # reference (and mutate under the optimizer). Reject it rather
            # than silently snapshotting a live graph.
            raise TypeError(
                "AsyncCheckpointSaver cannot snapshot an nn.Module; "
                "pass state_dict() / explicit tensors instead"
            )
        if isinstance(obj, dict):
            return {k: AsyncCheckpointSaver._snapshot(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(AsyncCheckpointSaver._snapshot(v) for v in obj)
        return obj

    @staticmethod
    def _atomic_save(obj, path):
        # Unique tmp name: up to max_inflight writes may be in flight at
        # once, and two saves can target the same path (e.g. consecutive
        # best-PSNR epochs). A fixed '.tmp' suffix would let concurrent
        # writers truncate each other's file.
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except BaseException:
            # A failed write must never leave a half-written .tmp behind:
            # a stale tmp would be silently overwritten by the next save and
            # masks the real failure on disk.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save(self, checkpoint, path):
        if self._closed:
            raise RuntimeError("AsyncCheckpointSaver is closed")
        snap = self._snapshot(checkpoint)
        self._pending.append(self._executor.submit(self._atomic_save, snap, str(path)))
        # Prune completed futures so the list cannot grow unboundedly over
        # a long run (one entry per save otherwise). Calling result() on the
        # finished ones surfaces any background write failure early instead
        # of hiding it until close().
        still_pending = []
        for fut in self._pending:
            if fut.done():
                fut.result()
            else:
                still_pending.append(fut)
        self._pending = still_pending

    def _drain(self, wait):
        still_pending = []
        error = None
        for fut in self._pending:
            if wait or fut.done():
                try:
                    fut.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
            else:
                still_pending.append(fut)
        # Cancel remaining in-flight futures so no write is silently lost
        # even when an earlier one already failed.
        for fut in still_pending:
            fut.cancel()
        self._pending = []
        if error is not None:
            raise error

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._drain(wait=True)
        finally:
            self._executor.shutdown(wait=True)


def parse_args():
    parser = argparse.ArgumentParser(description="MIM Training")
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "radar"]
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_length", type=int, default=20)
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="data loading workers; -1 = auto-tune from CPU count",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=-1,
        help="main-process intra-op CPU threads; -1 = auto "
        "(small pool when GPU training to leave cores "
        "to DataLoader workers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="training device (cuda requires GPU)",
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--ss_start_epoch", type=int, default=0)
    parser.add_argument("--ss_stop_epoch", type=int, default=50)
    parser.add_argument("--ss_initial_prob", type=float, default=1.0)
    parser.add_argument("--ss_final_prob", type=float, default=0.0)
    parser.add_argument(
        "--tln",
        dest="tln",
        action="store_true",
        default=True,
        help="enable Tensor Layer Norm",
    )
    parser.add_argument(
        "--no_tln", dest="tln", action="store_false", help="disable Tensor Layer Norm"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic mixed precision training",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="AMP dtype: float16 (all GPUs, needs GradScaler) "
        "or bfloat16 (Ampere+, numerically stabler)",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        default=False,
        help="use channels-last memory format for conv "
        "Tensor-Core fast paths (CUDA, experimental)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="enable torch.compile acceleration (experimental "
        "when combined with --grad_ckpt)",
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        default=False,
        help="temporal gradient checkpointing: ~1 extra "
        "forward pass per step in exchange for O(1) "
        "instead of O(T) activation memory",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        default=False,
        help="disable tqdm progress bar (tqdm remains a hard "
        "dependency; use this to keep CI logs clean)",
    )
    parser.add_argument(
        "--no_tensorboard",
        dest="tensorboard",
        action="store_false",
        default=True,
        help="disable TensorBoard logging (tqdm still active)",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard log directory; default: <save_dir>/tensorboard",
    )
    parser.add_argument(
        "--tensorboard_video_interval",
        type=int,
        default=5,
        help="write GT|Pred|Error video every N eval epochs",
    )
    parser.add_argument(
        "--tensorboard_video_samples",
        type=int,
        default=2,
        help="number of test samples per video (memory bound)",
    )
    parser.add_argument(
        "--tensorboard_model_graph",
        action="store_true",
        default=False,
        help="one-time model graph dump on first epoch "
        "(uses add_graph, which can fail on dynamic shapes)",
    )
    return parser.parse_args()


def _validate_args(args):
    """Fail fast on CLI values that would otherwise crash mid-run or no-op.

    Reads every field via ``getattr`` with a sane default so the validator is
    robust to partial ``argparse.Namespace`` objects (used by unit tests) and
    to future flag additions. Values not consumed here (e.g. ``data_path``)
    are validated by their own constructors.
    """
    log_interval = getattr(args, "log_interval", 100)
    save_interval = getattr(args, "save_interval", 5)
    batch_size = getattr(args, "batch_size", 8)
    epochs = getattr(args, "epochs", 100)
    lr = getattr(args, "lr", 1e-3)
    total_length = getattr(args, "total_length", 20)
    input_length = getattr(args, "input_length", 10)
    kernel_size = getattr(args, "kernel_size", 3)
    hidden_dim = getattr(args, "hidden_dim", [64, 64, 64, 64])
    num_workers = getattr(args, "num_workers", 0)
    num_threads = getattr(args, "num_threads", -1)
    prefetch_factor = getattr(args, "prefetch_factor", 2)

    if log_interval < 1:
        raise ValueError("--log_interval must be >= 1 to avoid modulo by zero")
    if save_interval < 1:
        raise ValueError("--save_interval must be >= 1 to avoid modulo by zero")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if lr <= 0:
        raise ValueError("--lr must be > 0")
    if total_length < 1:
        raise ValueError("--total_length must be >= 1")
    if input_length < 1:
        raise ValueError("--input_length must be >= 1")
    if total_length <= input_length:
        raise ValueError(
            f"--total_length ({total_length}) must be > --input_length ({input_length})"
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"--kernel_size must be a positive odd int, got {kernel_size}")
    if not hidden_dim or any(d < 1 for d in hidden_dim):
        raise ValueError(
            f"--hidden_dim must be a non-empty list of ints >= 1, got {hidden_dim}"
        )
    if num_workers < -1:
        raise ValueError("--num_workers must be -1 (auto) or >= 0")
    if num_threads < -1:
        raise ValueError("--num_threads must be -1 (auto) or >= 1")
    if prefetch_factor < 1:
        raise ValueError("--prefetch_factor must be >= 1")
    vid_int = getattr(args, "tensorboard_video_interval", 5)
    vid_n = getattr(args, "tensorboard_video_samples", 2)
    if vid_int < 1:
        raise ValueError("--tensorboard_video_interval must be >= 1")
    if vid_n < 1:
        raise ValueError("--tensorboard_video_samples must be >= 1")

    ss_initial = getattr(args, "ss_initial_prob", 1.0)
    ss_final = getattr(args, "ss_final_prob", 0.0)
    if not 0.0 <= ss_initial <= 1.0:
        raise ValueError(f"--ss_initial_prob must be in [0, 1], got {ss_initial}")
    if not 0.0 <= ss_final <= 1.0:
        raise ValueError(f"--ss_final_prob must be in [0, 1], got {ss_final}")
    ss_start = getattr(args, "ss_start_epoch", 0)
    ss_stop = getattr(args, "ss_stop_epoch", 0)
    if ss_start < 0 or ss_stop < 0:
        raise ValueError("--ss_start_epoch/--ss_stop_epoch must be >= 0")
    if ss_start >= ss_stop:
        raise ValueError(
            f"--ss_start_epoch ({ss_start}) must be < --ss_stop_epoch ({ss_stop})"
        )


def configure_runtime(args, device):
    """CPU thread partitioning and CUDA kernel-selection knobs."""
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    elif device.type == "cuda":
        # Keep the main-process intra-op pool small so DataLoader workers
        # (1 thread each) are not oversubscribed.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device.type == "cuda":
        # Fixed input shapes -> let cudnn autotune conv algorithms.
        torch.backends.cudnn.benchmark = True
        # TF32 for conv: 8x faster fp32 convs on Ampere+ with ample precision
        # for this task (separate from matmul TF32 below).
        torch.backends.cudnn.allow_tf32 = True
        # TF32 for fp32 matmuls: ~free speedup, ample precision.
        torch.set_float32_matmul_precision("high")


_RESUME_KEY_TYPES = {
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict,
    "epoch": int,
    "best_psnr": (int, float),
    "train_loss": (int, float),
    "metrics": dict,
    "args": dict,
}
_RESUME_REQUIRED_KEYS = ("model_state_dict", "optimizer_state_dict", "epoch")
_MAX_CKPT_BYTES = 5 * 1024**3  # 5 GiB defensive cap against zip-bomb DoS


def _load_resume_checkpoint(path, device, scaler):
    ckpt_size = os.path.getsize(str(path))
    if ckpt_size > _MAX_CKPT_BYTES:
        raise ValueError(
            f"resume checkpoint {path} is {ckpt_size} bytes, exceeding the "
            f"{_MAX_CKPT_BYTES}-byte safe-load limit; refusing to load a "
            f"potentially malicious oversized file"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"resume checkpoint must be a dict, got {type(checkpoint).__name__}"
        )
    for key in checkpoint:
        if key not in _RESUME_KEY_TYPES:
            raise ValueError(f"unexpected key {key!r} in resume checkpoint {path}")
    for key in _RESUME_REQUIRED_KEYS:
        if key not in checkpoint:
            raise ValueError(
                f"missing required key {key!r} in resume checkpoint {path}"
            )
    for key, expected in _RESUME_KEY_TYPES.items():
        if key in checkpoint:
            val = checkpoint[key]
            if isinstance(val, bool):
                raise ValueError(
                    f"resume checkpoint key {key!r} is bool — "
                    f"likely a stale CPU checkpoint; re-run or drop --resume"
                )
            if not isinstance(val, expected):
                raise ValueError(
                    f"resume checkpoint key {key!r} has wrong type: "
                    f"expected {expected}, got {type(val).__name__}"
                )
    if scaler is not None and "scaler_state_dict" not in checkpoint:
        scaler_state = getattr(scaler, "state_dict", lambda: {})()
        checkpoint["scaler_state_dict"] = scaler_state
    return checkpoint


def _check_resume_args(checkpoint, args):
    """Reject a resume whose architecture or sequence semantics differ.

    Weight shapes alone are not enough to catch every mismatch:
      * total_length / input_length live only in ``args`` metadata.
      * kernel_size / hidden_dim / tln determine which layer keys exist
        and the tensor shape, but the wrong CLI value can silently
        produce an *equally-shaped* yet *semantically different* load
        (e.g. hidden_dim=[32,32] vs [16,16,16,16] both sum to 64).
    Therefore we check every architecture-defining field that survives
    as a scalar or flag in the saved ``args`` namespace.
    """
    ckpt_args = checkpoint.get("args") or {}

    # Scalar fields whose mismatch would silently change training
    # semantics (re-sliced sequences, different hidden state layout,
    # or a TLN / no-TLN model loaded onto the wrong weights).
    for key in ("total_length", "input_length", "kernel_size", "tln"):
        ckpt_val = ckpt_args.get(key)
        cur_val = getattr(args, key, None)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            raise ValueError(
                f"resume checkpoint was trained with --{key}={ckpt_val} "
                f"but current CLI passes {cur_val}; architecture or "
                f"sequence semantics would silently change. Match the "
                f"value or drop --resume"
            )

    # hidden_dim may be a list in the checkpoint and a comma-separated
    # string at CLI-parse time. Normalise both before comparing so that
    # a [32,32] vs '32,32' round-trip does not spuriously reject a
    # legitimate resume.
    def _normalise_hidden(val):
        if isinstance(val, str):
            try:
                return [int(x.strip()) for x in val.split(",") if x.strip()]
            except ValueError:
                return val
        if isinstance(val, (list, tuple)):
            return list(val)
        return val

    ckpt_hidden = _normalise_hidden(ckpt_args.get("hidden_dim"))
    cur_hidden = _normalise_hidden(getattr(args, "hidden_dim", None))
    if ckpt_hidden is not None and cur_hidden is not None and ckpt_hidden != cur_hidden:
        raise ValueError(
            f"resume checkpoint was trained with --hidden_dim="
            f"{ckpt_args.get('hidden_dim')} but current CLI passes "
            f"{getattr(args, 'hidden_dim', None)}; hidden-state "
            f"layout would silently change. Match the value or "
            f"drop --resume"
        )


def get_scheduled_sampling_prob(epoch, args):
    if epoch < args.ss_start_epoch:
        return args.ss_initial_prob
    if epoch >= args.ss_stop_epoch:
        return args.ss_final_prob
    progress = (epoch - args.ss_start_epoch) / max(
        1, args.ss_stop_epoch - args.ss_start_epoch
    )
    prob = args.ss_initial_prob - progress * (args.ss_initial_prob - args.ss_final_prob)
    # Clamp: a misconfigured ``ss_start > ss_stop`` would otherwise yield
    # negative probabilities that silently disable teacher forcing.
    return min(1.0, max(0.0, prob))


def generate_ss_bool(
    batch_size, total_length, input_length, height, width, prob, device
):
    ss_length = total_length - input_length - 1
    if ss_length < 0:
        raise ValueError(
            f"total_length ({total_length}) must be > input_length "
            f"({input_length}) to yield a non-negative scheduled-sampling "
            f"length; got ss_length={ss_length}"
        )
    ss_bool = (
        torch.rand(batch_size, ss_length, height, width, device=device) < prob
    ).float()
    return ss_bool


def _check_grad_norm_finite(grad_norm, epoch, batch_idx):
    """Guard the non-GradScaler path: a NaN/Inf gradient total-norm means
    clipping was a no-op and ``optimizer.step()`` would write NaN into the
    weights (GradScaler.step skips such steps; bf16/plain-fp32 paths have
    no such guard without this check).
    """
    finite = torch.isfinite(grad_norm)
    if torch.is_tensor(finite):
        finite = finite.item()
    if not finite:
        raise FloatingPointError(
            f"non-finite gradient norm at epoch {epoch}, batch {batch_idx} "
            f"({grad_norm!r}); aborting before weights are corrupted"
        )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    epoch,
    args,
    device,
    scaler=None,
    amp_dtype=None,
    tb_logger=None,
):
    model.train()
    use_amp = amp_dtype is not None and device.type == "cuda"
    # Accumulate loss as a fp32 tensor: one device sync per log interval
    # instead of one per batch, and immune to fp16 precision loss in the
    # running average over an entire epoch.
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    num_batches = 0
    last_loss = 0.0

    ss_prob = get_scheduled_sampling_prob(epoch, args)
    disable_bar = getattr(args, "no_progress_bar", False)

    with EpochProgress(
        total=len(dataloader), desc=f"Epoch {epoch}", disable=disable_bar
    ) as bar:
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device, non_blocking=True)
            batch_size = frames.shape[0]

            if ss_prob <= 0.0:
                # Pure autoregressive: ss=0 everywhere, identical to None.
                ss_bool = None
            else:
                ss_bool = generate_ss_bool(
                    batch_size,
                    args.total_length,
                    args.input_length,
                    frames.shape[3],
                    frames.shape[4],
                    ss_prob,
                    device,
                )

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    gen_imgs = model(frames, ss_bool)
                    target = frames[:, 1:]
                    loss = criterion(gen_imgs, target)
            else:
                gen_imgs = model(frames, ss_bool)
                target = frames[:, 1:]
                loss = criterion(gen_imgs, target)

            if not torch.isfinite(loss):
                # Check BEFORE backward/step: once we call .backward() on a
                # NaN loss the gradients are NaN, and optimizer.step() would
                # silently propagate NaN into the model weights, corrupting
                # the in-memory best before we can save. Raising here keeps
                # both the weights and any in-flight AsyncCheckpointSaver
                # snapshots clean.
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, batch {batch_idx} "
                    f"({loss.item()!r}); aborting before weights are "
                    f"corrupted. Last healthy checkpoint is preserved."
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                _check_grad_norm_finite(grad_norm, epoch, batch_idx)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                _check_grad_norm_finite(grad_norm, epoch, batch_idx)
                optimizer.step()

            total_loss += loss.detach().float()
            num_batches += 1
            last_loss = float(loss.item())
            avg_loss = float((total_loss / num_batches).item())
            bar.update(
                1, loss=f"{last_loss:.4f}", avg=f"{avg_loss:.4f}", ss=f"{ss_prob:.2f}"
            )

    if num_batches == 0:
        raise ValueError(
            f"epoch {epoch} produced no batches: dataset has fewer samples "
            f"than batch_size with drop_last=True; reduce --batch_size"
        )

    epoch_loss = (total_loss / num_batches).item()
    if tb_logger is not None:
        # step = total batches seen so far (epoch * num_batches) gives a
        # monotonically increasing x-axis even if eval cadence is sparse.
        step = epoch * num_batches + num_batches
        tb_logger.log_train_loss(epoch_loss, step)
        tb_logger.log_lr(optimizer.param_groups[0]["lr"], step)
    return epoch_loss


@torch.inference_mode()
def evaluate(model, dataloader, args, device, max_video_samples=0):
    """Sample-weighted metrics with a single device sync at the end.

    Uses ``inference_mode`` (not ``no_grad``): disables version counters
    and view tracking on top of grad tracking for a small extra speedup.
    The dataloader must NOT use drop_last so every test sample is scored.

    If ``max_video_samples > 0`` the first batch's predictions are also
    returned (clipped to ``max_video_samples`` rows) so the caller can
    write a 3-row GT|Pred|Error video to TensorBoard.
    """
    model.eval()
    if getattr(dataloader, "drop_last", False):
        raise ValueError(
            "evaluate() requires drop_last=False so every test sample is "
            "scored; a drop_last loader silently drops the tail batch and "
            "biases the metrics"
        )
    sums = {
        k: torch.zeros((), device=device, dtype=torch.float32)
        for k in ("mse", "psnr", "ssim", "mae")
    }
    total = 0
    skipped_nan = 0
    captured = None

    for frames in dataloader:
        frames = frames.to(device, non_blocking=True)
        gen_imgs = model(frames, ss_bool=None)

        target = frames[:, args.input_length :]
        pred = gen_imgs[:, args.input_length - 1 :]
        if pred.shape != target.shape:
            raise ValueError(
                f"eval misalignment: pred {tuple(pred.shape)} vs target "
                f"{tuple(target.shape)} (check input_length/total_length)"
            )

        b = frames.shape[0]
        batch_mse_v = batch_mse(pred, target)
        batch_psnr_v = batch_psnr(pred, target)
        batch_ssim_v = batch_ssim(pred, target)
        batch_mae_v = batch_mae(pred, target)

        # batch_* metrics are batch-averaged scalars. If a single sample in
        # the batch was corrupted (NaN from a broken forward, a degenerate
        # SSIM window, a fp16 overflow in AMP), the scalar will be non-
        # finite. Skip the whole batch rather than letting the NaN silently
        # poison the running sums for the rest of the epoch.
        if not (
            torch.isfinite(batch_mse_v)
            and torch.isfinite(batch_psnr_v)
            and torch.isfinite(batch_ssim_v)
            and torch.isfinite(batch_mae_v)
        ):
            skipped_nan += b
            continue

        sums["mse"] += batch_mse_v * b
        sums["psnr"] += batch_psnr_v * b
        sums["ssim"] += batch_ssim_v * b
        sums["mae"] += batch_mae_v * b
        total += b

        if captured is None and max_video_samples > 0:
            n = min(max_video_samples, pred.shape[0])
            captured = (pred[:n].clamp(0, 1).cpu(), target[:n].clamp(0, 1).cpu())

    metrics = {k: (v / max(total, 1)).item() for k, v in sums.items()}
    metrics["samples"] = total
    metrics["skipped_nan"] = skipped_nan
    if total == 0 and skipped_nan > 0:
        raise FloatingPointError(
            f"evaluate: all {skipped_nan} samples produced non-finite "
            f"metrics; model weights may be corrupted"
        )
    if captured is not None:
        return metrics, captured
    return metrics


def main():
    args = parse_args()
    _validate_args(args)

    os.makedirs(args.save_dir, exist_ok=True)

    # Respect an explicit --device cpu; only fall back when the requested
    # cuda device is unavailable.
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if args.device != "cpu" and device.type == "cpu":
        print(
            f"Warning: --device {args.device} requested but CUDA is not "
            f"available; falling back to CPU (training will be much slower)"
        )
    print(f"Using device: {device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    configure_runtime(args, device)

    if args.num_workers < 0:
        args.num_workers = recommend_num_workers(device.type == "cuda")
        print(f"Auto-tuned num_workers: {args.num_workers}")

    if args.dataset == "mnist":
        train_dataset = MovingMNIST(
            args.data_path, args.total_length, args.input_length, is_train=True
        )
        test_dataset = MovingMNIST(
            args.data_path, args.total_length, args.input_length, is_train=False
        )
    else:
        train_dataset = RadarEcho(
            args.data_path, args.total_length, args.input_length, is_train=True
        )
        test_dataset = RadarEcho(
            args.data_path, args.total_length, args.input_length, is_train=False
        )
    img_channel, img_height, img_width = train_dataset.frame_shape

    train_loader = get_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    # drop_last=False: every test sample must be scored.
    test_loader = get_dataloader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    in_shape = [args.batch_size, img_channel, img_height, img_width]
    model = MIM(
        input_dims=img_channel,
        out_dims=img_channel,
        in_shape=in_shape,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        total_length=args.total_length,
        input_length=args.input_length,
        tln=args.tln,
    ).to(device)

    if getattr(args, "channels_last", False) and device.type == "cuda":
        model.channels_last = True
        model = model.to(memory_format=torch.channels_last)
        print("Channels-last enabled (conv weights + per-step activation layout)")

    if args.grad_ckpt:
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled (temporal loop)")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, fused=(device.type == "cuda")
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    scaler = None
    amp_dtype = None
    if args.amp and device.type == "cuda":
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")
            print("AMP enabled (float16 + GradScaler)")
        else:
            print("AMP enabled (bfloat16, no GradScaler)")
    elif args.amp and device.type != "cuda":
        print("Warning: --amp requires CUDA; ignoring.")

    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        checkpoint = _load_resume_checkpoint(args.resume, device, scaler)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)
        _check_resume_args(checkpoint, args)
        if start_epoch >= args.epochs:
            raise ValueError(
                f"resume start_epoch {start_epoch} >= --epochs {args.epochs}; "
                f"increase --epochs or drop --resume"
            )
        print(f"Resumed from epoch {start_epoch} (best PSNR {best_psnr:.2f})")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # TensorBoard logger: created after --tensorboard is finalised so a
    # misconfigured flag fails before we touch the model.
    tb_logger = None
    if args.tensorboard:
        tb_dir = args.tensorboard_dir or os.path.join(args.save_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        try:
            tb_logger = TBLogger(tb_dir)
        except ImportError as exc:
            print(f"Error: {exc}")
            sys.exit(2)
        print(
            f"TensorBoard logging to {tb_dir}; launch with: "
            f"tensorboard --logdir {tb_dir}"
        )
        if args.tensorboard_model_graph:
            with torch.no_grad():
                example = torch.zeros(
                    1,
                    args.total_length,
                    img_channel,
                    img_height,
                    img_width,
                    device=device,
                )
            tb_logger.log_model_graph(model, example)

    saver = AsyncCheckpointSaver()
    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                epoch,
                args,
                device,
                scaler=scaler,
                amp_dtype=amp_dtype,
                tb_logger=tb_logger,
            )
            scheduler.step()

            print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}")

            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                capture = (
                    tb_logger is not None
                    and (epoch + 1) % args.tensorboard_video_interval == 0
                )
                result = evaluate(
                    model,
                    test_loader,
                    args,
                    device,
                    max_video_samples=args.tensorboard_video_samples if capture else 0,
                )
                if capture:
                    metrics, (pred_vid, target_vid) = result
                else:
                    metrics = result
                print(
                    f"Epoch {epoch} - Eval: "
                    f"MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}, "
                    f"SSIM={metrics['ssim']:.4f}, MAE={metrics['mae']:.6f}"
                )

                if tb_logger is not None:
                    tb_logger.log_eval_metrics(metrics, epoch)
                    if capture:
                        tb_logger.log_eval_video(
                            pred_vid, target_vid, epoch, max_samples=pred_vid.shape[0]
                        )

                model_to_save = getattr(model, "_orig_mod", model)
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "metrics": metrics,
                    "best_psnr": best_psnr,
                    "args": vars(args),
                }
                if scaler is not None:
                    checkpoint["scaler_state_dict"] = scaler.state_dict()

                # Save best checkpoint BEFORE epoch checkpoint so a crash
                # between the two leaves mim_best.pth consistent with the
                # in-memory best_psnr (not the stale value from the prior
                # save window).
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    checkpoint["best_psnr"] = best_psnr
                    best_path = os.path.join(args.save_dir, "mim_best.pth")
                    saver.save(checkpoint, best_path)
                    print(f"New best PSNR: {best_psnr:.2f}")

                save_path = os.path.join(args.save_dir, f"mim_epoch_{epoch}.pth")
                saver.save(checkpoint, save_path)
    finally:
        # Drain any in-flight checkpoint writes even on Ctrl-C / exception.
        saver.close()
        if tb_logger is not None:
            tb_logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
