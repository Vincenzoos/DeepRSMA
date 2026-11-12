"""Train the DNA-adapted DeepRSMA baseline on Thesis A's shared AptaBench folds."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import main_cv as upstream_cv
from data.aptabench_adapter import (
    AptaBenchAdapter,
    GB_DNA_HIDDEN_SIZE,
    GB_DNA_MODEL,
    GB_DNA_REPOSITORY,
    GB_DNA_REVISION,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
    materialize_gbdna_embeddings,
)

_THESIS_ROOT = Path(__file__).resolve().parents[2]
_THESIS_SRC = _THESIS_ROOT / "src"
if str(_THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(_THESIS_SRC))

from affinity_experiment.metrics import calculate_metrics  # noqa: E402

_STRATEGIES = ("unseen_pair", "unseen_aptamer", "unseen_molecule", "unseen_both")
_BASE_SEED = 42
_DEFAULT_OUTPUT = (
    _THESIS_ROOT
    / "outputs"
    / "sota_comparison"
    / "deeprsma"
    / "aptabench"
    / "cross_validation"
)
_DEFAULT_EMBEDDINGS = (
    _THESIS_ROOT / "outputs" / "sota_comparison" / "deeprsma" / "aptabench" / "gbdna"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"deeprsma.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train" / "train.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _make_run_dir(output_root: Path, strategy: str, fold_id: int, seed: int) -> Path:
    strategy_root = output_root / strategy
    strategy_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-fold{fold_id}-s{seed}"
    run_dir = strategy_root / run_id
    if run_dir.exists():
        run_dir = strategy_root / f"{run_id}-{uuid.uuid4().hex[:8]}"
    (run_dir / "train").mkdir(parents=True, exist_ok=False)
    (run_dir / "eval").mkdir(parents=True, exist_ok=False)
    return run_dir


def _dataset_loader(
    examples: list[tuple[Any, Any]], *, batch_size: int, num_workers: int
) -> DataLoader:
    rna_items = [pair[0] for pair in examples]
    molecule_items = [pair[1] for pair in examples]
    dataset = upstream_cv.CustomDualDataset(rna_items, molecule_items)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    batch_limit: int | None = None,
) -> tuple[list[float], list[float]]:
    targets: list[float] = []
    predictions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (rna_batch, molecule_batch) in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            scores = model(rna_batch.to(device), molecule_batch.to(device)).reshape(-1)
            targets.extend(rna_batch.y.detach().cpu().reshape(-1).tolist())
            predictions.extend(scores.detach().cpu().tolist())
    return targets, predictions


def _pearson(targets: Iterable[float], predictions: Iterable[float]) -> float | None:
    target_values = np.asarray(list(targets), dtype=np.float64)
    prediction_values = np.asarray(list(predictions), dtype=np.float64)
    if len(target_values) < 2 or np.ptp(target_values) == 0 or np.ptp(prediction_values) == 0:
        return None
    value = float(pearsonr(target_values, prediction_values).statistic)
    return value if math.isfinite(value) else None


def _regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, Any]:
    size = len(targets)
    metrics = calculate_metrics(
        binding_targets=np.zeros(size, dtype=np.float32),
        binding_available=np.zeros(size, dtype=bool),
        binding_probabilities=np.zeros(size, dtype=np.float32),
        pkd_targets=np.asarray(targets, dtype=np.float32),
        pkd_available=np.ones(size, dtype=bool),
        pkd_predictions=np.asarray(predictions, dtype=np.float32),
    )
    return metrics["regression"]


def _evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fold_id: int,
    test_rows: tuple[Any, ...],
    run_dir: Path,
) -> dict[str, Any]:
    logger = logging.getLogger(f"deeprsma.{run_dir.name}.eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(run_dir / "eval" / "evaluate.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    targets, predictions = _predict(model, loader, device)
    if len(targets) != len(test_rows):
        raise RuntimeError(
            f"test loader emitted {len(targets)} rows, expected {len(test_rows)} pKd rows"
        )
    metrics = _regression_metrics(targets, predictions)
    prediction_path = run_dir / "eval" / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "fold_id",
                "example_id",
                "aptamer_id",
                "molecule_id",
                "pKd_value",
                "prediction",
            ),
        )
        writer.writeheader()
        for row, target, prediction in zip(test_rows, targets, predictions):
            writer.writerow(
                {
                    "fold_id": fold_id,
                    "example_id": row.example_id,
                    "aptamer_id": row.aptamer_id,
                    "molecule_id": row.molecule_id,
                    "pKd_value": target,
                    "prediction": prediction,
                }
            )
    _write_json(run_dir / "eval" / "test_metrics.json", {"regression": metrics})
    logger.info("test pKd rows=%d metrics=%s", len(targets), json.dumps(metrics, sort_keys=True))
    return metrics


def run_fold(
    *,
    strategy: str,
    fold_id: int,
    embedding_dir: Path,
    output_root: Path,
    device: torch.device,
    epochs: int = 1500,
    batch_size: int = 16,
    num_workers: int = 1,
    smoke: bool = False,
    smoke_examples: int = 4,
) -> dict[str, Any]:
    """Train and test one fold; smoke mode caps each partition and runs one epoch."""

    seed = _BASE_SEED + fold_id
    upstream_cv.set_seed(seed)
    upstream_cv.device = device
    adapter = AptaBenchAdapter(strategy, embedding_dir)
    partition, datasets = adapter.datasets_for_fold(
        fold_id,
        limit_per_partition=smoke_examples if smoke else None,
    )
    resolved_epochs = 1 if smoke else epochs
    resolved_workers = 0 if smoke else num_workers
    run_dir = _make_run_dir(output_root, strategy, fold_id, seed)
    logger = _make_logger(run_dir)
    train_rows = partition.train[:smoke_examples] if smoke else partition.train
    val_rows = partition.val[:smoke_examples] if smoke else partition.val
    test_rows = partition.test[:smoke_examples] if smoke else partition.test
    config = {
        "dataset": "AptaBench",
        "split_mode": "cross_validation",
        "split_strategy": strategy,
        "fold_id": fold_id,
        "base_seed": _BASE_SEED,
        "resolved_seed": seed,
        "epochs": resolved_epochs,
        "batch_size": batch_size,
        "num_workers": resolved_workers,
        "smoke_run": smoke,
        "source_url": UPSTREAM_URL,
        "source_commit": UPSTREAM_COMMIT,
        "dna_structure_input": "Thesis A ViennaRNA DNA BPP graph; adjacency only",
        "dna_sequence_model": GB_DNA_MODEL,
        "dna_sequence_repository": GB_DNA_REPOSITORY,
        "dna_sequence_revision": GB_DNA_REVISION,
        "dna_embedding_width": GB_DNA_HIDDEN_SIZE,
        "assignment_counts_before_pkd_filter": dict(partition.assigned_counts),
        "pkd_available_counts": dict(partition.pkd_counts),
        "rows_used": {
            "train": len(datasets["train"]),
            "val": len(datasets["val"]),
            "test": len(datasets["test"]),
        },
        "checkpoint_selection": (
            "maximum validation Pearson correlation; test rows used once after restore"
        ),
    }
    _write_json(run_dir / "config.json", config)

    train_loader = _dataset_loader(
        datasets["train"], batch_size=batch_size, num_workers=resolved_workers
    )
    val_loader = _dataset_loader(
        datasets["val"], batch_size=batch_size, num_workers=resolved_workers
    )
    test_loader = _dataset_loader(
        datasets["test"], batch_size=batch_size, num_workers=resolved_workers
    )
    model = upstream_cv.DeepRSMA(rna_embedding_dim=GB_DNA_HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-7)
    loss_function = torch.nn.MSELoss()
    checkpoint_path = run_dir / "best_model.pt"
    best_pcc: float | None = None
    best_val_mse = float("inf")
    best_epoch = -1

    logger.info(
        "start strategy=%s fold=%d seed=%d train=%d val=%d test=%d assigned=%s pKd=%s device=%s",
        strategy,
        fold_id,
        seed,
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        dict(partition.assigned_counts),
        dict(partition.pkd_counts),
        device,
    )
    for epoch in tqdm(range(resolved_epochs), desc=f"{strategy} fold {fold_id}", unit="epoch"):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_rna, batch_molecule in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_rna.to(device), batch_molecule.to(device)).reshape(-1)
            loss = loss_function(predictions, batch_rna.y)
            loss.backward()
            optimizer.step()
            batch_size_actual = int(batch_rna.y.shape[0])
            total_loss += float(loss.detach()) * batch_size_actual
            total_examples += batch_size_actual

        # Preserve the upstream max-Pearson criterion while keeping held-out test rows unused.
        val_targets, val_predictions = _predict(model, val_loader, device)
        if len(val_targets) != len(datasets["val"]):
            raise RuntimeError("validation loader did not preserve all pKd-available rows")
        val_mse = float(np.mean(np.square(np.asarray(val_targets) - np.asarray(val_predictions))))
        val_pcc = _pearson(val_targets, val_predictions)
        improved = (
            best_epoch < 0
            or (val_pcc is not None and (best_pcc is None or val_pcc > best_pcc))
            or (val_pcc is None and best_pcc is None and val_mse < best_val_mse)
        )
        logger.info(
            "epoch=%d train_mse=%.8f val_mse=%.8f val_pcc=%s",
            epoch + 1,
            total_loss / max(total_examples, 1),
            val_mse,
            "undefined" if val_pcc is None else f"{val_pcc:.6f}",
        )
        if improved:
            best_pcc = val_pcc
            best_val_mse = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    if best_epoch < 1 or not checkpoint_path.is_file():
        raise RuntimeError(f"no validation checkpoint was selected for {strategy} fold {fold_id}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    test_metrics = _evaluate_test(model, test_loader, device, fold_id, test_rows, run_dir)
    result = {
        "strategy": strategy,
        "fold_id": fold_id,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_validation_pcc": best_pcc,
        "test_regression": test_metrics,
    }
    _write_json(run_dir / "result.json", result)
    logger.info("complete result=%s", json.dumps(result, sort_keys=True))
    return result


def _summary(results: list[dict[str, Any]], strategy: str, output_root: Path) -> Path:
    metrics: dict[str, dict[str, Any]] = {}
    for name in ("r2", "rmse", "mae", "spearman"):
        values = [
            float(item["test_regression"][name])
            for item in results
            if item["test_regression"].get(name) is not None
            and math.isfinite(float(item["test_regression"][name]))
        ]
        metrics[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
            "defined_fold_count": len(values),
        }
    payload = {
        "split_strategy": strategy,
        "fold_count": len(results),
        "regression": metrics,
        "fold_results": results,
    }
    summary_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_id += f"-{uuid.uuid4().hex[:8]}"
    path = output_root / strategy / f"aggregate-{summary_id}" / "summary.json"
    _write_json(path, payload)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=_STRATEGIES, default="unseen_pair")
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--fold-id", type=int, help="run one zero-based fold")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--embedding-dir", type=Path, default=_DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--prepare-dna-embeddings", action="store_true")
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one epoch and at most four rows per partition",
    )
    parser.add_argument("--smoke-examples", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.smoke_examples < 2:
        raise SystemExit("smoke-examples must be at least 2")
    if args.all_strategies and args.fold_id is not None:
        raise SystemExit("--fold-id cannot be combined with --all-strategies")

    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {device} is not available")

    if args.prepare_dna_embeddings:
        manifest = materialize_gbdna_embeddings(
            args.embedding_dir,
            device=device,
            overwrite=args.overwrite_embeddings,
        )
        print(
            f"Prepared {len(manifest['entries'])} GB.DNA-300M aptamer embeddings at "
            f"{args.embedding_dir}"
        )
        return 0
    if args.overwrite_embeddings:
        raise SystemExit("--overwrite-embeddings requires --prepare-dna-embeddings")
    if args.all_strategies and args.smoke:
        raise SystemExit("--smoke is a one-fold check and cannot be combined with --all-strategies")
    if args.smoke and args.fold_id is None:
        raise SystemExit("--smoke requires --fold-id so it runs exactly one fold")

    output_root = args.output_root.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    strategies = _STRATEGIES if args.all_strategies else (args.strategy,)
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        fold_ids = (args.fold_id,) if args.fold_id is not None else (0, 1, 2, 3, 4)
        strategy_results = []
        for fold_id in fold_ids:
            result = run_fold(
                strategy=strategy,
                fold_id=fold_id,
                embedding_dir=embedding_dir,
                output_root=output_root,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                smoke=args.smoke,
                smoke_examples=args.smoke_examples,
            )
            strategy_results.append(result)
            results.append(result)
        if len(strategy_results) == 5:
            summary_path = _summary(strategy_results, strategy, output_root)
            print(f"Wrote five-fold summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
