"""Re-evaluate one completed DeepRSMA AptaBench cross-validation fold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import main_aptabench_cv as runner
import main_cv as upstream_cv
from data.aptabench_adapter import AptaBenchAdapter, GB_DNA_HIDDEN_SIZE


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return payload


def _resolve_device(value: str) -> torch.device:
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {device} is not available")
    return device


def _refresh_aggregate(run_dir: Path, result: dict[str, Any]) -> None:
    strategy_root = run_dir.parent
    for summary_path in strategy_root.glob("aggregate-*/summary.json"):
        summary = _read_json(summary_path)
        fold_results = summary.get("fold_results")
        if not isinstance(fold_results, list):
            continue
        match_index = next(
            (
                index
                for index, item in enumerate(fold_results)
                if isinstance(item, dict)
                and Path(str(item.get("run_dir", ""))).expanduser().resolve() == run_dir
            ),
            None,
        )
        if match_index is None:
            continue
        fold_results[match_index] = result
        metrics: dict[str, dict[str, Any]] = {}
        for name in ("r2", "rmse", "mae", "spearman"):
            values = [
                float(item["test_regression"][name])
                for item in fold_results
                if item.get("test_regression", {}).get(name) is not None
                and math.isfinite(float(item["test_regression"][name]))
            ]
            metrics[name] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
                "defined_fold_count": len(values),
            }
        summary["regression"] = metrics
        runner._write_json(summary_path, summary)


def evaluate(
    *,
    dataset_dir: Path,
    run_dir: Path,
    device: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    result_path = run_dir / "result.json"
    result = _read_json(result_path)
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"selected checkpoint is missing: {checkpoint_path}")

    strategy = config.get("split_strategy")
    fold_id = config.get("fold_id")
    identity = config.get("dataset_identity")
    if config.get("dataset") != "AptaBench" or not isinstance(identity, dict):
        raise SystemExit(
            "run config lacks AptaBench dataset identity; train with the current runner "
            "before using standalone evaluation"
        )
    if result.get("strategy") != strategy or result.get("fold_id") != fold_id:
        raise SystemExit("result.json does not match the training configuration")
    recorded_run_dir = result.get("run_dir")
    if not isinstance(recorded_run_dir, str) or Path(recorded_run_dir).resolve() != run_dir:
        raise SystemExit("result.json run_dir does not match --run-dir")

    embedding_dir_value = config.get("embedding_dir")
    if not isinstance(embedding_dir_value, str):
        raise SystemExit("run config is missing embedding_dir")
    adapter = AptaBenchAdapter(strategy, dataset_dir, Path(embedding_dir_value))
    split = adapter.split
    expected_identity = {
        "split_mode": split.split_mode,
        "split_strategy": split.split_strategy,
        "source_fingerprint": split.source_fingerprint,
        "fold_ids": list(split.fold_ids),
    }
    if identity != expected_identity:
        raise SystemExit(
            "--dataset-dir split mode, strategy, source fingerprint, or fold set does not "
            "match the training run"
        )
    if fold_id not in split.fold_ids:
        raise SystemExit(f"fold {fold_id!r} is not present in --dataset-dir")

    smoke = bool(config.get("smoke_run", False))
    smoke_examples = int(config.get("smoke_examples", 4))
    limit = smoke_examples if smoke else None
    partition, datasets = adapter.datasets_for_fold(fold_id, limit_per_partition=limit)
    test_rows = partition.test[:limit] if limit is not None else partition.test
    rows_used = config.get("rows_used")
    expected_rows = rows_used.get("test") if isinstance(rows_used, dict) else None
    if expected_rows != len(test_rows):
        raise SystemExit(
            f"test rows from --dataset-dir ({len(test_rows)}) do not match the run config "
            f"({expected_rows})"
        )

    train_batch_size = int(config.get("batch_size", 16))
    eval_batch_size = batch_size or train_batch_size
    resolved_workers = num_workers
    if resolved_workers is None:
        resolved_workers = int(config.get("num_workers", 1))
    if resolved_workers < 0:
        raise SystemExit("num-workers cannot be negative")
    if eval_batch_size < 1:
        raise SystemExit("batch-size must be positive")
    device_value = _resolve_device(device)
    upstream_cv.set_seed(int(config.get("resolved_seed", 42 + fold_id)))
    upstream_cv.device = device_value
    loader = runner._dataset_loader(
        datasets["test"], batch_size=eval_batch_size, num_workers=resolved_workers
    )
    model = upstream_cv.DeepRSMA(rna_embedding_dim=GB_DNA_HIDDEN_SIZE).to(device_value)
    try:
        state = torch.load(checkpoint_path, map_location=device_value, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        state = torch.load(checkpoint_path, map_location=device_value)
    model.load_state_dict(state, strict=True)

    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    metrics = runner._evaluate_test(model, loader, device_value, fold_id, test_rows, run_dir)
    result["test_regression"] = metrics
    result["run_dir"] = str(run_dir)
    runner._write_json(result_path, result)
    _refresh_aggregate(run_dir, result)
    print(json.dumps({"run_dir": str(run_dir), "test_regression": metrics}, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="processed cross-validation strategy directory used for training",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="one completed fold run directory containing config.json and best_model.pt",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--num-workers", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_workers is not None and args.num_workers < 0:
        raise SystemExit("num-workers cannot be negative")
    evaluate(
        dataset_dir=args.dataset_dir,
        run_dir=args.run_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
