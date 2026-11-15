"""Pre-training loop (Algorithm 1): sample prior datasets, NLL on interventional targets, AdamW.

Seeds (decision D6): training batch ``step`` uses datasets ``derive_seed(seed, step, b)``; the
validation set and the prior samples for the bin borders come from their own streams
(``VAL_STREAM``, ``BORDER_STREAM``), so they never overlap with training data. Everything a step
does is determined by ``(seed, step)``, so data workers and resuming give the same run.
"""

import csv
import math
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dopfn.common.config import load_yaml
from dopfn.common.seed import set_seed
from dopfn.data.prior import PriorConfig, PriorSampler, derive_seed, load_prior_config
from dopfn.model.adapter import ModelBatch, Target, to_model_batch
from dopfn.model.bar_distribution import make_borders
from dopfn.model.loss import nll_loss
from dopfn.model.transformer import Model, ModelConfig, normalize_y, y_stats

VAL_STREAM = 1
BORDER_STREAM = 2
CHECKPOINT_LATEST = "checkpoint_latest.pt"
CHECKPOINT_BEST = "checkpoint_best.pt"
METRICS_FILE = "metrics.csv"
METRIC_FIELDS = (
    "step", "lr", "minutes", "train_nll", "val_nll", "val_outcome_mse", "val_cate_mse",
    "val_cate_mse_group_means",
)


@dataclass
class TrainConfig:
    """Training hyperparameters, filled from the ``train`` section of ``configs/train/*.yaml``.

    Attributes:
        steps: Optimizer steps.
        batch_size: Datasets per step.
        lr: Peak learning rate (AdamW).
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup, then cosine decay to 0.
        grad_clip: Max gradient norm.
        n_train_min: Smallest number of observational rows (sampled per step).
        n_train_max: Largest number of observational rows.
        n_test: Query rows per dataset.
        target: ``"interventional"`` (Do-PFN) or ``"observational"`` (sanity check, Dont-PFN).
        val_every: Validate every this many steps.
        n_val_datasets: Size of the fixed validation set.
        val_batch_size: Datasets per validation batch (one row count per batch).
        n_border_datasets: Prior datasets used to place the bin borders.
        log_every: Average the training loss over this many steps per log line.
        checkpoint_every: Save ``checkpoint_latest.pt`` every this many steps.
        num_workers: Data-generating worker processes (0: main process).
        max_minutes: Stop cleanly after this wall time (before a SLURM limit); ``None``: no limit.
        out_dir: Output directory for metrics, checkpoints and the loss curve.
        seed: Base seed.
        device: ``"auto"`` (CUDA if available, else CPU), ``"cuda"``, ``"mps"`` or ``"cpu"``.
        prior: Prior config.
    """

    steps: int = 10_000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip: float = 1.0
    n_train_min: int = 50
    n_train_max: int = 1000
    n_test: int = 200
    target: Target = "interventional"
    val_every: int = 500
    n_val_datasets: int = 256
    val_batch_size: int = 32
    n_border_datasets: int = 500
    log_every: int = 50
    checkpoint_every: int = 1000
    num_workers: int = 0
    max_minutes: float | None = None
    out_dir: str = "outputs/train"
    seed: int = 0
    device: str = "auto"
    prior: PriorConfig = field(default_factory=PriorConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.n_train_min <= self.n_train_max:
            raise ValueError("Need 1 <= n_train_min <= n_train_max.")
        if self.target not in ("interventional", "observational"):
            raise ValueError(f"Unknown target {self.target!r}")


def load_train_config(path: str | Path, **overrides: Any) -> tuple[ModelConfig, TrainConfig]:
    """Load ``configs/train/*.yaml`` into a ``ModelConfig`` and a ``TrainConfig``.

    Layout: top-level ``seed``, ``device``, ``data`` (path to a prior config YAML, relative to the
    working directory), and the sections ``model`` and ``train``.

    Args:
        path: YAML file.
        overrides: ``TrainConfig`` fields that replace the file's values (``None`` values ignored).

    Returns:
        Model and training config.
    """
    raw = load_yaml(path)
    unknown = set(raw) - {"seed", "device", "data", "model", "train"}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    train_kwargs = dict(raw.get("train") or {})
    for key in ("seed", "device"):
        if key in raw:
            train_kwargs[key] = raw[key]
    if raw.get("data"):
        train_kwargs["prior"] = load_prior_config(raw["data"])
    train_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**(raw.get("model") or {})), TrainConfig(**train_kwargs)


def resolve_device(device: str) -> torch.device:
    """``"auto"`` picks CUDA if available, else the CPU.

    Apple MPS is only used when asked for explicitly: the number of rows changes every step, and
    MPS rebuilds its compute graph for every new shape, which made training stall in our tests.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sample_batch(
    sampler: PriorSampler, seed: int, step: int, config: TrainConfig
) -> ModelBatch:
    """The training batch of ``step`` (CPU), fully determined by ``(seed, step)``."""
    generator = torch.Generator().manual_seed(derive_seed(seed, step))
    n_train = int(torch.randint(config.n_train_min, config.n_train_max + 1, (1,), generator=generator))
    samples = [
        sampler.sample(derive_seed(seed, step, b), n_train=n_train, n_test=config.n_test).data
        for b in range(config.batch_size)
    ]
    return to_model_batch(samples, target=config.target)


class PriorBatches(IterableDataset):
    """Training batches for steps ``start, ..., stop - 1``, in order, split over workers."""

    def __init__(self, config: TrainConfig, start: int, stop: int) -> None:
        super().__init__()
        self.config, self.start, self.stop = config, start, stop

    def __iter__(self) -> Iterator[tuple[int, ModelBatch]]:
        info = get_worker_info()
        worker, n_workers = (0, 1) if info is None else (info.id, info.num_workers)
        sampler = PriorSampler(self.config.prior)
        # DataLoader reads workers round-robin, so step order is preserved.
        for step in range(self.start + worker, self.stop, n_workers):
            yield step, sample_batch(sampler, self.config.seed, step, self.config)


def validation_batches(config: TrainConfig) -> list[ModelBatch]:
    """Fixed validation set from its own seed stream; one row count per batch."""
    sampler = PriorSampler(config.prior)
    batches = []
    n_batches = math.ceil(config.n_val_datasets / config.val_batch_size)
    for k in range(n_batches):
        generator = torch.Generator().manual_seed(derive_seed(VAL_STREAM, k))
        n_train = int(torch.randint(config.n_train_min, config.n_train_max + 1, (1,), generator=generator))
        size = min(config.val_batch_size, config.n_val_datasets - k * config.val_batch_size)
        samples = [
            sampler.sample(derive_seed(VAL_STREAM, k, b), n_train=n_train, n_test=config.n_test).data
            for b in range(size)
        ]
        batches.append(to_model_batch(samples, target=config.target))
    return batches


def compute_borders(config: TrainConfig, n_bins: int) -> torch.Tensor:
    """Bin borders from normalized targets of prior datasets (own seed stream)."""
    sampler = PriorSampler(config.prior)
    targets = []
    for i in range(config.n_border_datasets):
        size_gen = torch.Generator().manual_seed(derive_seed(BORDER_STREAM, i))
        n_train = int(torch.randint(config.n_train_min, config.n_train_max + 1, (1,), generator=size_gen))
        batch = to_model_batch(
            [sampler.sample(derive_seed(BORDER_STREAM, i, 0), n_train=n_train, n_test=config.n_test).data],
            target=config.target,
        )
        targets.append(normalize_y(batch.y_target, batch.y_ob).flatten())
    return make_borders(n_bins, torch.cat(targets))


@torch.no_grad()
def evaluate(model: Model, batches: list[ModelBatch], device: torch.device) -> dict[str, float]:
    """Validation metrics, all on the normalized outcome scale (comparable across datasets).

    Returns:
        ``val_nll``; ``val_outcome_mse`` of the predicted mean; ``val_cate_mse`` of the predicted
        CATE; ``val_cate_mse_group_means``: the same for the difference of group means (reference
        for G2).
    """
    was_training = model.training
    model.eval()
    totals = {"val_nll": 0.0, "val_outcome_mse": 0.0, "val_cate_mse": 0.0, "val_cate_mse_group_means": 0.0}
    n_rows = 0
    for batch in batches:
        X, y_ob = batch.X.to(device), batch.y_ob.to(device)
        y_target = normalize_y(batch.y_target.to(device), y_ob)
        cate = batch.cate.to(device) / y_stats(y_ob)[1]
        logits = model(X, y_ob)
        rows = y_target.numel()
        totals["val_nll"] += nll_loss(logits, y_target, model.bar).item() * rows
        totals["val_outcome_mse"] += ((model.bar.mean(logits) - y_target) ** 2).sum().item()

        X1, X0 = X.clone(), X.clone()
        X1[:, batch.n_ob :, 0], X0[:, batch.n_ob :, 0] = 1.0, 0.0
        cate_pred = model.bar.mean(model(X1, y_ob)) - model.bar.mean(model(X0, y_ob))
        totals["val_cate_mse"] += ((cate_pred - cate) ** 2).sum().item()

        t_ob, y_norm = X[:, : batch.n_ob, 0], normalize_y(y_ob, y_ob)
        n1, n0 = t_ob.sum(1), (1 - t_ob).sum(1)
        group_diff = (y_norm * t_ob).sum(1) / n1.clamp_min(1) - (y_norm * (1 - t_ob)).sum(1) / n0.clamp_min(1)
        group_diff = torch.where((n1 > 0) & (n0 > 0), group_diff, torch.zeros_like(group_diff))
        totals["val_cate_mse_group_means"] += ((group_diff.unsqueeze(1) - cate) ** 2).sum().item()
        n_rows += rows
    model.train(was_training)
    return {k: v / n_rows for k, v in totals.items()}


def _git_commit() -> str:
    """Current git commit, or ``"unknown"``."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lr_factor(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay to 0."""
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _config_dict(config: TrainConfig) -> dict[str, Any]:
    """Checkpoint-safe (primitive) view of a ``TrainConfig``, including the prior config."""
    d = asdict(config)
    d["prior"]["treated_fraction"] = list(d["prior"]["treated_fraction"])
    return d


def save_checkpoint(
    model: Model,
    path: str | Path,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: TrainConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model config, weights (incl. bin borders) and training state to ``path``."""
    state = {
        "step": step,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "train_config": _config_dict(config) if config is not None else None,
        "git_commit": _git_commit(),
        **(extra or {}),
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a job killed while saving keeps the previous checkpoint


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Model:
    """Load a model (with its bin borders) from a checkpoint written by ``save_checkpoint``."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    model = Model.from_config(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.eval()


def _write_metrics(out_dir: Path, row: dict[str, Any]) -> None:
    """Append one row to ``metrics.csv`` (header on first write)."""
    path = out_dir / METRICS_FILE
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def read_metrics(out_dir: str | Path) -> list[dict[str, float]]:
    """Rows of ``metrics.csv`` with empty cells dropped."""
    with (Path(out_dir) / METRICS_FILE).open() as f:
        return [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]


def plot_loss_curve(out_dir: str | Path) -> Path:
    """Plot training and validation NLL from ``metrics.csv`` to ``loss_curve.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("train_nll", "training"), ("val_nll", "validation")):
        points = [(r["step"], r[key]) for r in rows if key in r]
        if points:
            ax.plot(*zip(*points, strict=True), label=label, marker="o" if key == "val_nll" else None, markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("NLL (normalized outcome)")
    ax.set_title("Pre-training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train(model: Model, config: TrainConfig, resume: bool = False) -> dict[str, list[float]]:
    """Pre-train ``model`` on prior data.

    Logs training and validation metrics to ``metrics.csv``, writes ``checkpoint_latest.pt``,
    ``checkpoint_best.pt`` (lowest validation NLL) and ``loss_curve.png`` to ``config.out_dir``.

    Args:
        model: Model to train (modified in place).
        config: Training config.
        resume: Continue from ``checkpoint_latest.pt`` in ``out_dir`` if it exists.

    Returns:
        Loss history ``{"step": [...], "train": [...], "val_step": [...], "val": [...]}`` of this call.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_factor(s, config))

    start_step, best_val = 0, math.inf
    latest = out_dir / CHECKPOINT_LATEST
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.to(device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, best_val = state["step"], state.get("best_val", math.inf)
        print(f"Resuming from step {start_step}")
    else:
        if (out_dir / METRICS_FILE).exists():
            (out_dir / METRICS_FILE).unlink()
        model.set_borders(compute_borders(config, model.config.n_bins))
    model.to(device).train()

    val_batches = validation_batches(config)
    history: dict[str, list[float]] = {"step": [], "train": [], "val_step": [], "val": []}
    loader = DataLoader(
        PriorBatches(config, start_step, config.steps),
        batch_size=None,
        num_workers=config.num_workers,
        persistent_workers=False,
    )

    def checkpoint(path: Path, step: int) -> None:
        save_checkpoint(model, path, step, optimizer, scheduler, config, extra={"best_val": best_val})

    t0, running, n_running, step = time.time(), 0.0, 0, start_step
    for step, batch in loader:
        X, y_ob = batch.X.to(device), batch.y_ob.to(device)
        y_target = normalize_y(batch.y_target.to(device), y_ob)
        loss = nll_loss(model(X, y_ob), y_target, model.bar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        running, n_running = running + loss.item(), n_running + 1

        done = step + 1
        minutes = (time.time() - t0) / 60
        row: dict[str, Any] = {}
        if done % config.log_every == 0 or done == config.steps:
            row.update(train_nll=running / n_running)
            history["step"].append(done)
            history["train"].append(running / n_running)
            running, n_running = 0.0, 0
        if done % config.val_every == 0 or done == config.steps:
            metrics = evaluate(model, val_batches, device)
            row.update(metrics)
            history["val_step"].append(done)
            history["val"].append(metrics["val_nll"])
            if metrics["val_nll"] < best_val:
                best_val = metrics["val_nll"]
                checkpoint(out_dir / CHECKPOINT_BEST, done)
        if row:
            row.update(step=done, lr=scheduler.get_last_lr()[0], minutes=round(minutes, 2))
            _write_metrics(out_dir, row)
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if done % config.checkpoint_every == 0 or done == config.steps:
            checkpoint(latest, done)
        if config.max_minutes is not None and minutes >= config.max_minutes:
            checkpoint(latest, done)
            print(f"Stopping after {minutes:.1f} min at step {done} (max_minutes).", flush=True)
            break

    if (out_dir / METRICS_FILE).exists():
        plot_loss_curve(out_dir)
    return history
