"""MIM inference: rolling autoregressive prediction with optional CUDA Graphs.

Usage:
    python inference.py --checkpoint checkpoints/mim_best.pth \
        --input frames.npy --input_length 10 --horizon 10 \
        --use_cuda_graph --output pred.npy

CUDA Graphs remove per-kernel launch overhead and re-planning cost by
capturing the whole fixed-shape forward pass once and replaying it. They
require static shapes/addresses and an RNG-free model, both of which hold
for MIM in eval mode (see PyTorch CUDA semantics docs).
"""

import argparse
import os
import threading
import time

import numpy as np
import torch

from mim import MIM

# Protects in-place mutation of model.total_length inside predict(). MIM is
# not inherently thread-safe: the forward pass reads self.total_length while
# the rolling loop writes it. Re-entrant calls on the SAME model from
# different threads would otherwise observe a torn total_length.
_predict_lock = threading.RLock()

# Defensive resource-exhaustion bounds for externally-supplied inputs.
# ``horizon`` comes from the inference CLI/API and directly multiplies both
# the padded tensor size and the autoregressive loop length; ``_MAX_CKPT_BYTES``
# bounds a checkpoint file that ``torch.load(weights_only=True)`` would
# otherwise deserialize eagerly (a zip-bomb-style oversized tensor OOM).
_MAX_HORIZON = 100_000
_MAX_CKPT_BYTES = 5 * 1024**3  # 5 GiB


def infer_architecture(state_dict, height=None, width=None):
    """Recover (input_dims, hidden_dim, kernel_size, height, width, tln).

    ``height``/``width`` override the values inferred from the weights;
    they are required when the checkpoint has a single layer (no
    convlstm_c / ct_weight tensors to recover spatial size from).
    """
    if "stlstm_layer.0.x_cc.weight" not in state_dict:
        raise ValueError("state_dict does not look like a MIM checkpoint")
    x_cc = state_dict["stlstm_layer.0.x_cc.weight"]  # [4*h0, in, k, k]
    if not torch.is_tensor(x_cc) or x_cc.ndim != 4:
        raise ValueError(
            f"'stlstm_layer.0.x_cc.weight' must be a 4-D conv weight "
            f"[out, in, k, k], got {getattr(x_cc, 'shape', type(x_cc).__name__)}"
        )
    if x_cc.shape[-1] != x_cc.shape[-2]:
        raise ValueError(
            f"'stlstm_layer.0.x_cc.weight' must have square kernels, "
            f"got shape {tuple(x_cc.shape)}"
        )
    if x_cc.shape[-1] < 1 or x_cc.shape[1] < 1 or x_cc.shape[0] < 1:
        raise ValueError(
            f"'stlstm_layer.0.x_cc.weight' has degenerate dims {tuple(x_cc.shape)}"
        )
    input_dims = int(x_cc.shape[1])
    kernel_size = int(x_cc.shape[-1])

    hidden_dim = []
    i = 0
    while f"stlstm_layer.{i}.x_cc.weight" in state_dict:
        w = state_dict[f"stlstm_layer.{i}.x_cc.weight"]
        if not torch.is_tensor(w) or w.ndim != 4:
            raise ValueError(
                f"state_dict key 'stlstm_layer.{i}.x_cc.weight' must be a "
                f"4-D conv weight, got {getattr(w, 'shape', type(w).__name__)}"
            )
        out_ch = int(w.shape[0])
        if out_ch % 4 != 0:
            raise ValueError(
                f"state_dict key 'stlstm_layer.{i}.x_cc.weight' has "
                f"{out_ch} output channels, not a multiple of 4; the "
                f"checkpoint is malformed or not a MIM checkpoint"
            )
        hidden_dim.append(out_ch // 4)
        i += 1

    # Height/width are only recoverable from convlstm_c tensors (H, W
    # spatial dims), which exist from layer 1 on. Try several possible
    # key variants across different training versions.
    ct = None
    for key in (
        "stlstm_layer.1.mims.ct_weight",
        "stlstm_layer.1.convlstm_c_ct_weight",
        "stlstm_layer.1.mimn.ct_weight",
        "stlstm_layer_diff.0.ct_weight",
    ):
        if key in state_dict:
            ct = state_dict[key]
            break
    if ct is not None:
        height, width = int(ct.shape[-2]), int(ct.shape[-1])
    elif height is None or width is None:
        raise ValueError(
            "cannot infer H/W from a single-layer checkpoint; pass "
            "--height/--width explicitly"
        )
    if height < 1 or width < 1:
        raise ValueError(f"height/width must be >= 1, got ({height}, {width})")
    tln = "stlstm_layer.0.tln_t.gamma" in state_dict
    return input_dims, hidden_dim, kernel_size, height, width, tln


def load_model(
    checkpoint_path,
    device="cuda",
    total_length=20,
    input_length=10,
    height=None,
    width=None,
):
    """Build a MIM from a checkpoint, inferring the architecture from weights.

    ``height``/``width`` override the inferred spatial size; they are
    required for single-layer checkpoints where H/W cannot be recovered
    from the weights.
    """
    ckpt_size = os.path.getsize(str(checkpoint_path))
    if ckpt_size > _MAX_CKPT_BYTES:
        raise ValueError(
            f"checkpoint {checkpoint_path} is {ckpt_size} bytes, exceeding "
            f"the {_MAX_CKPT_BYTES}-byte safe-load limit; refusing to load "
            f"a potentially malicious oversized file"
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    input_dims, hidden_dim, kernel_size, ckpt_h, ckpt_w, tln = infer_architecture(
        state_dict, height=height, width=width
    )

    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_args, dict):
        ckpt_il = ckpt_args.get("input_length")
        if ckpt_il is not None and ckpt_il != input_length:
            raise ValueError(
                f"checkpoint was trained with input_length={ckpt_il} "
                f"but loading with input_length={input_length}; "
                f"model architecture and temporal semantics would "
                f"silently change. Match the value or edit the "
                f"checkpoint args."
            )

    model = MIM(
        input_dims=input_dims,
        out_dims=input_dims,
        in_shape=[1, input_dims, ckpt_h, ckpt_w],
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        total_length=total_length,
        input_length=input_length,
        tln=tln,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(torch.device(device))


def pad_frames(frames, total_length):
    """Repeat the last frame until T >= total_length.

    Frames at t >= input_length are never consumed by the model in pure
    autoregressive mode (ss_bool=0), so padding content is irrelevant.
    """
    if frames.shape[1] >= total_length:
        return frames
    pad = total_length - frames.shape[1]
    last = frames[:, -1:].expand(-1, pad, -1, -1, -1)
    return torch.cat([frames, last], dim=1)


@torch.inference_mode()
def predict(model, frames, horizon):
    """Rolling prediction: given [B, T_in, C, H, W], predict `horizon` frames.

    Returns [B, horizon, C, H, W]. The model's total_length is temporarily
    set to input_length + horizon; the padded tail frames are ignored.

    The whole call is guarded by a module-level lock because the rolling
    loop mutates ``model.total_length`` in-place. Without the lock, two
    threads calling ``predict`` on the same model could observe each
    other's total_length mid-forward, producing silently wrong outputs or
    a RuntimeError under ``torch.compile``.
    """
    if frames.dim() != 5:
        raise ValueError(f"expected [B, T, C, H, W], got {tuple(frames.shape)}")
    if hasattr(model, "_orig_mod"):
        # A torch.compile wrapper shadows attribute mutation: setting
        # model.total_length would not reliably reach the inner module's
        # forward, producing stale-graph or torn-read behaviour.
        raise ValueError(
            "predict() does not support a torch.compile-wrapped model; "
            "call predict(model._orig_mod, ...) or use the uncompiled model"
        )
    if frames.shape[1] < model.input_length:
        raise ValueError(
            f"need >= {model.input_length} input frames, got {frames.shape[1]}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if horizon > _MAX_HORIZON:
        raise ValueError(
            f"horizon {horizon} exceeds the safe limit {_MAX_HORIZON}; "
            f"this would pad and roll the model to an unreasonable length"
        )

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if frames.device != device or frames.dtype != model_dtype:
        frames = frames.to(device=device, dtype=model_dtype)

    required_steps = model.input_length + horizon
    with _predict_lock:
        old_total = model.total_length
        model.total_length = required_steps
        try:
            padded = pad_frames(frames, model.total_length)
            gen = model(padded, ss_bool=None)
            return gen[:, model.input_length - 1 :]
        finally:
            model.total_length = old_total


class CUDAGraphRunner:
    """Capture one fixed-shape inference call into a CUDA graph and replay it.

    Requirements (PyTorch CUDA semantics): static input/output shapes and
    memory addresses, warm-up on a side stream before capture, no RNG or
    host-side mutation between replays. Shape/device mismatches fall back
    to eager execution.
    """

    def __init__(self, model, example_input, warmup_iters=3):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphRunner requires a CUDA device")
        if example_input.device.type != "cuda":
            raise ValueError("example_input must live on the GPU")
        model.eval()
        self.model = model
        self.static_input = example_input.clone()
        self._lock = threading.Lock()

        # Free cached-but-unused allocator blocks so graph capture does not
        # unnecessarily hold onto fragmented segments.
        torch.cuda.empty_cache()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(warmup_iters):
                self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.no_grad(), torch.cuda.graph(self.graph):
                self.static_output = self.model(self.static_input)
        except Exception as exc:
            # Capture can fail (OOM, unsupported op, allocator state). Degrade
            # to eager execution instead of crashing the whole inference run.
            self.graph = None
            self.static_input = None
            print(
                f"Warning: CUDA graph capture failed ({exc}); "
                f"falling back to eager execution"
            )

    def __call__(self, frames):
        if self.graph is None:
            with torch.no_grad():
                return self.model(frames)
        if (
            tuple(frames.shape) != tuple(self.static_input.shape)
            or frames.device != self.static_input.device
            or frames.dtype != self.static_input.dtype
            or not frames.is_contiguous()
            or bool(frames.requires_grad)
        ):
            with torch.no_grad():
                return self.model(frames)
        with self._lock:
            self.static_input.copy_(frames)
            self.graph.replay()
            return self.static_output.clone()


def benchmark(model, frames, use_cuda_graph=False, repeats=20, runner=None):
    """Mean latency per forward call in milliseconds.

    Pass an already-constructed ``runner`` to avoid capturing a second CUDA
    graph (graph capture is expensive and re-capturing the same model wastes
    device memory).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if use_cuda_graph:
        call = runner if runner is not None else CUDAGraphRunner(model, frames)
    else:

        def call(f):
            return model(f)

    device = frames.device
    with torch.no_grad():
        for _ in range(3):
            call(frames)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            call(frames)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def load_input(path, input_length):
    """Load [T,H,W] / [T,C,H,W] / [B,T,C,H,W] .npy into [B,T,C,H,W] float.

    Accepts ``uint8`` (normalized to [0,1]) and float arrays. Other dtypes
    (uint16/int32/etc.) are rejected so a silent misinterpretation of a
    raw sensor dump as pixel data is impossible.
    """
    arr = np.load(str(path))
    if arr.ndim == 3:  # [T, H, W]
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 or arr.ndim == 5:  # [T, C, H, W]
        pass
    else:
        raise ValueError(f"unsupported input ndim {arr.ndim}")
    n_frames = arr.shape[-4] if arr.ndim == 5 else arr.shape[0]
    if n_frames < input_length:
        raise ValueError(f"need >= {input_length} frames, got shape {arr.shape}")
    if arr.dtype not in (np.uint8, np.float16, np.float32, np.float64):
        raise ValueError(
            f"unsupported input dtype {arr.dtype}; expected uint8 or float"
        )
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    frames = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return frames.unsqueeze(0) if frames.dim() == 4 else frames


def main():
    parser = argparse.ArgumentParser(description="MIM Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=".npy with [T,H,W], [T,C,H,W] or [B,T,C,H,W]",
    )
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="override spatial height (required for single-layer checkpoints)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="override spatial width (required for single-layer checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to run inference on",
    )
    parser.add_argument("--use_cuda_graph", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="inference precision; fp16 roughly halves device "
        "memory and bandwidth usage (CUDA only)",
    )
    parser.add_argument("--output", type=str, default="pred.npy")
    args = parser.parse_args()
    if args.input_length < 1:
        parser.error("--input_length must be >= 1")
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    frames = load_input(args.input, args.input_length)

    model = load_model(
        args.checkpoint,
        device=device,
        total_length=args.input_length + args.horizon,
        input_length=args.input_length,
        height=args.height,
        width=args.width,
    )
    if args.precision == "float16":
        if device.type != "cuda":
            print("Warning: fp16 inference requires CUDA; keeping float32")
        else:
            model = model.half()
            frames = frames.half()
            print("Inference precision: float16")
    print(
        f"Loaded {args.checkpoint} on {device} "
        f"(input_length={model.input_length}, horizon={args.horizon})"
    )

    if args.use_cuda_graph and device.type == "cuda":
        padded = pad_frames(frames.to(device), model.total_length)
        runner = CUDAGraphRunner(model, padded)
        if args.benchmark:
            eager_ms = benchmark(model, padded, use_cuda_graph=False)
            graph_ms = benchmark(model, padded, use_cuda_graph=True, runner=runner)
            print(
                f"Latency: eager {eager_ms:.2f} ms | "
                f"CUDA graph {graph_ms:.2f} ms "
                f"({eager_ms / max(graph_ms, 1e-9):.2f}x)"
            )
        pred = runner(padded)[:, model.input_length - 1 :]
    else:
        if args.use_cuda_graph:
            print("CUDA not available; falling back to eager execution")
        if args.benchmark:
            ms = benchmark(model, pad_frames(frames.to(device), model.total_length))
            print(f"Latency: eager {ms:.2f} ms")
        pred = predict(model, frames, args.horizon)

    pred = pred.clamp(0, 1).cpu().numpy()
    np.save(args.output, pred)
    print(f"Saved predictions {pred.shape} to {args.output}")


if __name__ == "__main__":
    main()
