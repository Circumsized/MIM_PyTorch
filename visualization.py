"""Training-time visualization: tqdm progress and TensorBoard logging.

TensorBoard scalar/video APIs are imported lazily so that the rest of the
training pipeline stays usable when the ``tensorboard`` package is missing.
tqdm is imported eagerly because the user requested it be always enabled.
"""

import torch
import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
    _TB_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised in the missing-deps test
    _TB_IMPORT_ERROR = e
    SummaryWriter = None
    _TB_AVAILABLE = False


def make_video_grid(gt, pred):
    """Stack [B, T, 1, H, W] GT/Pred/abs-error into [B, T, 1, 3H, W].

    The third row is a grayscale error heatmap where intensity is
    proportional to the per-pixel |GT - Pred| (zero = black, max = white).
    Layout is GT | Pred | Error so a TB reader can scan top-to-bottom and
    see the model output, then the failure mode, in one image.
    """
    if gt.shape != pred.shape:
        raise ValueError(
            f"gt/pred shape mismatch: {tuple(gt.shape)} vs {tuple(pred.shape)}"
        )
    err = (gt - pred).abs()
    max_err = err.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    err_gray = (err / max_err).clamp(0, 1)
    return torch.cat([gt, pred, err_gray], dim=3)


class TBLogger:
    """Thin wrapper around ``SummaryWriter`` that fails fast on missing dep.

    Use ``is_available()`` to gate UI features without paying the import
    cost in code paths that never log (e.g. inference).
    """

    def __init__(self, log_dir):
        if not _TB_AVAILABLE:
            raise ImportError(
                f"tensorboard is required for --tensorboard but is not "
                f"installed: {_TB_IMPORT_ERROR}. Run "
                f"`pip install tensorboard` and retry."
            )
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._closed = False

    def log_train_loss(self, loss, step):
        self._writer.add_scalar("train/loss", float(loss), step)

    def log_lr(self, lr, step):
        self._writer.add_scalar("train/lr", float(lr), step)

    def log_eval_metrics(self, metrics, epoch):
        for k, v in metrics.items():
            self._writer.add_scalar(f"eval/{k}", float(v), epoch)

    def log_eval_video(self, pred, target, epoch, max_samples=2):
        """Write a 3-row video (GT | Pred | error) for a few test samples."""
        n = min(max_samples, pred.shape[0])
        if n == 0:
            return
        grid = make_video_grid(target[:n], pred[:n])
        # TB expects [N, T, C, H, W] float in [0, 1]; already satisfied.
        self._writer.add_video(f"eval/prediction_epoch_{epoch}", grid, 1)

    def log_model_graph(self, model, example_input):
        try:
            self._writer.add_graph(model, example_input)
        except Exception as exc:
            # add_graph fails on dynamic-shape forward; never block training.
            self._writer.add_text(
                "model/graph_skip",
                f"add_graph unavailable: {type(exc).__name__}: {exc}",
            )

    def close(self):
        if self._closed:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True

    @staticmethod
    def is_available():
        return _TB_AVAILABLE


class EpochProgress:
    """tqdm-based progress bar with on-update loss/lr display.

    ``disable=True`` is supported so CI / log files can fall back to the
    non-interactive ``print`` path.
    """

    def __init__(self, total, desc, disable=False):
        self._bar = tqdm.tqdm(
            total=total,
            desc=desc,
            disable=disable,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{eta}, "
            "{rate_fmt}{postfix}]",
        )
        self._postfix = {}

    def update(self, n=1, **postfix):
        if postfix:
            self._postfix.update(
                {k: v for k, v in postfix.items() if isinstance(v, (int, float, str))}
            )
            self._bar.set_postfix(self._postfix)
        self._bar.update(n)

    def close(self):
        self._bar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
