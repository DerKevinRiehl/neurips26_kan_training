"""Visualize and summarize experiment 1 cluster results.

The raw result CSVs can come from several cluster runs. This script reads every
CSV in ``results/experiment_1_feynman``, ignores the trailing parameter blocks,
concatenates the data rows, deduplicates exact experiment/checkpoint keys, and
writes exploratory plots plus summary tables.

The main comparison uses the final checkpoint and chooses the best architecture
within each model type for every Feynman function and training-sample size.
That is useful for exploring sample efficiency, but it is still test-set model
selection rather than a validation-set estimate.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ###########################################################################
# 1. Paths and constants
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "experiment_1_feynman"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "visualizations"

PARAMETER_SEPARATOR = "####### PARAMETERS #######"
DEFAULT_METRIC = "test_eval_nrmse"

METRIC_COLUMNS = [
    "test_eval_mse",
    "test_eval_rmse",
    "test_eval_mae",
    "test_eval_nrmse",
    "test_eval_relative_l2",
    "test_eval_r2",
    "test_eval_max_absolute_error",
]

RESULT_KEY_COLUMNS = [
    "random_seed",
    "feynman_function",
    "model_setup_id",
    "model_type",
    "train_sample_size",
    "train_steps",
]

MODEL_LABELS = {
    "kan_bspline": "KAN B-spline",
    "kan_gaussrbf": "KAN Gaussian RBF",
    "mlp_relu": "MLP ReLU",
    "mlp_sigmoid": "MLP sigmoid",
}

MODEL_COLORS = {
    "kan_bspline": "#2563eb",
    "kan_gaussrbf": "#0f766e",
    "mlp_relu": "#dc2626",
    "mlp_sigmoid": "#9333ea",
}

MODEL_MARKERS = {
    "kan_bspline": "o",
    "kan_gaussrbf": "s",
    "mlp_relu": "^",
    "mlp_sigmoid": "D",
}


# ###########################################################################
# 2. Data model and loading
# ###########################################################################


@dataclass(slots=True)
class ResultRow:
    random_seed: int
    feynman_function: str
    model_setup_id: str
    model_type: str
    model_dims: str
    model_n_parameters: int
    train_sample_size: int
    train_steps: int
    train_loss: float
    test_loss: float
    runtime_seconds: float
    test_eval_mse: float
    test_eval_rmse: float
    test_eval_mae: float
    test_eval_nrmse: float
    test_eval_relative_l2: float
    test_eval_r2: float
    test_eval_max_absolute_error: float
    source_file: str

    @property
    def key(self) -> tuple[int, str, str, str, int, int]:
        return (
            self.random_seed,
            self.feynman_function,
            self.model_setup_id,
            self.model_type,
            self.train_sample_size,
            self.train_steps,
        )

    @property
    def job_key(self) -> tuple[int, str, str, str, int]:
        return (
            self.random_seed,
            self.feynman_function,
            self.model_setup_id,
            self.model_type,
            self.train_sample_size,
        )

    @property
    def setup_key(self) -> tuple[str, str, int, str]:
        return (
            self.model_type,
            self.feynman_function,
            self.train_sample_size,
            self.model_setup_id,
        )

    @property
    def model_family(self) -> str:
        if self.model_type.startswith("kan"):
            return "kan"
        if self.model_type.startswith("mlp"):
            return "mlp"
        return self.model_type.split("_", maxsplit=1)[0]

    def value(self, column: str) -> float:
        try:
            return float(getattr(self, column))
        except AttributeError as exc:
            raise KeyError(f"Unknown result column {column!r}.") from exc


@dataclass(slots=True)
class LoadReport:
    csv_paths: list[Path]
    raw_rows: int
    rows_after_deduplication: int

    @property
    def duplicate_rows(self) -> int:
        return self.raw_rows - self.rows_after_deduplication


def metric_lower_is_better(metric: str) -> bool:
    return metric != "test_eval_r2"


def model_label(model_type: str) -> str:
    return MODEL_LABELS.get(model_type, model_type)


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def iter_csv_data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith(PARAMETER_SEPARATOR):
                break
            yield line


def parse_result_row(row: dict[str, str], source_file: str) -> ResultRow:
    return ResultRow(
        random_seed=parse_int(row.get("random_seed")),
        feynman_function=str(row.get("feynman_function", "")),
        model_setup_id=str(row.get("model_setup_id", "")),
        model_type=str(row.get("model_type", "")),
        model_dims=str(row.get("model_dims", "")),
        model_n_parameters=parse_int(row.get("model_n_parameters")),
        train_sample_size=parse_int(row.get("train_sample_size")),
        train_steps=parse_int(row.get("train_steps")),
        train_loss=parse_float(row.get("train_loss")),
        test_loss=parse_float(row.get("test_loss")),
        runtime_seconds=parse_float(row.get("runtime_seconds")),
        test_eval_mse=parse_float(row.get("test_eval_mse")),
        test_eval_rmse=parse_float(row.get("test_eval_rmse")),
        test_eval_mae=parse_float(row.get("test_eval_mae")),
        test_eval_nrmse=parse_float(row.get("test_eval_nrmse")),
        test_eval_relative_l2=parse_float(row.get("test_eval_relative_l2")),
        test_eval_r2=parse_float(row.get("test_eval_r2")),
        test_eval_max_absolute_error=parse_float(
            row.get("test_eval_max_absolute_error")
        ),
        source_file=source_file,
    )


def load_result_csv(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    reader = csv.DictReader(iter_csv_data_lines(path))
    for raw_row in reader:
        if not raw_row.get("random_seed"):
            continue
        rows.append(parse_result_row(raw_row, path.name))
    return rows


def sort_result_row(row: ResultRow) -> tuple[object, ...]:
    return (
        row.feynman_function,
        row.train_sample_size,
        row.model_type,
        row.model_setup_id,
        row.random_seed,
        row.train_steps,
    )


def load_all_results(results_dir: Path, csv_glob: str) -> tuple[list[ResultRow], LoadReport]:
    csv_paths = sorted(results_dir.glob(csv_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir} matching {csv_glob!r}.")

    rows_by_key: dict[tuple[int, str, str, str, int, int], ResultRow] = {}
    raw_rows = 0
    for path in csv_paths:
        for row in load_result_csv(path):
            raw_rows += 1
            rows_by_key[row.key] = row

    rows = sorted(rows_by_key.values(), key=sort_result_row)
    report = LoadReport(
        csv_paths=csv_paths,
        raw_rows=raw_rows,
        rows_after_deduplication=len(rows),
    )
    return rows, report


# ###########################################################################
# 3. Summary helpers
# ###########################################################################


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    array = finite_array(values)
    if array.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std = 0.0 if array.size == 1 else float(array.std(ddof=1))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def group_rows(
    rows: Iterable[ResultRow],
    key_fn: Callable[[ResultRow], tuple[object, ...]],
) -> dict[tuple[object, ...], list[ResultRow]]:
    grouped: dict[tuple[object, ...], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def value_sort_key(value: object) -> tuple[int, float, str]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value), str(value))
    text = str(value)
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, 0.0, text)


def format_joined(values: Iterable[object], max_items: int = 12) -> str:
    sorted_values = [str(value) for value in sorted(set(values), key=value_sort_key)]
    if len(sorted_values) <= max_items:
        return ",".join(sorted_values)
    head = ",".join(sorted_values[:max_items])
    return f"{head},...(+{len(sorted_values) - max_items})"


def rounded(value: object, digits: int = 6) -> object:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return round(float(value), digits)
    return value


def rounded_row(row: dict[str, object]) -> dict[str, object]:
    return {key: rounded(value) for key, value in row.items()}


def write_csv_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rounded_row(row) for row in rows)


# ###########################################################################
# 4. Analysis tables
# ###########################################################################


def coverage_table(rows: list[ResultRow]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for (model_type,), group in sorted(
        group_rows(rows, lambda row: (row.model_type,)).items()
    ):
        table.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "rows": len(group),
                "feynman_functions": len({row.feynman_function for row in group}),
                "random_seeds": len({row.random_seed for row in group}),
                "model_setups": len({row.model_setup_id for row in group}),
                "min_parameters": min(row.model_n_parameters for row in group),
                "max_parameters": max(row.model_n_parameters for row in group),
                "train_sample_sizes": format_joined(row.train_sample_size for row in group),
                "train_steps": format_joined(row.train_steps for row in group),
                "source_files": format_joined(row.source_file for row in group),
            }
        )
    return table


def latest_rows_by_job(rows: list[ResultRow]) -> list[ResultRow]:
    latest: dict[tuple[int, str, str, str, int], ResultRow] = {}
    for row in rows:
        if row.job_key not in latest or row.train_steps > latest[row.job_key].train_steps:
            latest[row.job_key] = row
    return list(latest.values())


def final_checkpoint_rows(
    rows: list[ResultRow],
    final_step: int | None,
) -> tuple[list[ResultRow], int]:
    if final_step is None:
        final_step = max(row.train_steps for row in rows)
    final_rows = [row for row in rows if row.train_steps == final_step]
    if final_rows:
        return final_rows, final_step
    return latest_rows_by_job(rows), final_step


def setup_sort_key(
    setup_row: dict[str, object],
    *,
    metric: str,
    lower_is_better: bool,
) -> tuple[float, float, int, str]:
    median = float(setup_row[f"{metric}_median"])
    q75 = float(setup_row[f"{metric}_q75"])
    n_parameters = int(setup_row["model_n_parameters"])
    model_setup_id = str(setup_row["model_setup_id"])
    if lower_is_better:
        return median, q75, n_parameters, model_setup_id
    return -median, -q75, n_parameters, model_setup_id


def select_best_setups(
    final_rows: list[ResultRow],
    metric: str,
) -> tuple[list[ResultRow], list[dict[str, object]], dict[tuple[str, str, int], str]]:
    lower_is_better = metric_lower_is_better(metric)
    grouped = group_rows(
        final_rows,
        lambda row: (
            row.model_type,
            row.feynman_function,
            row.train_sample_size,
            row.model_setup_id,
        ),
    )

    setup_rows: list[dict[str, object]] = []
    for (model_type, function_name, sample_size, setup_id), group in grouped.items():
        stats = summarize_values(row.value(metric) for row in group)
        setup_rows.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "feynman_function": function_name,
                "train_sample_size": sample_size,
                "model_setup_id": setup_id,
                "model_n_parameters": int(np.median([row.model_n_parameters for row in group])),
                "seed_rows": stats["n"],
                f"{metric}_median": stats["median"],
                f"{metric}_q25": stats["q25"],
                f"{metric}_q75": stats["q75"],
                f"{metric}_mean": stats["mean"],
                f"{metric}_std": stats["std"],
            }
        )

    candidates_by_comparison: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in setup_rows:
        key = (
            str(row["model_type"]),
            str(row["feynman_function"]),
            int(row["train_sample_size"]),
        )
        candidates_by_comparison[key].append(row)

    best_setup_by_comparison: dict[tuple[str, str, int], str] = {}
    best_setup_rows: list[dict[str, object]] = []
    for key, candidates in candidates_by_comparison.items():
        best = sorted(
            candidates,
            key=lambda row: setup_sort_key(
                row,
                metric=metric,
                lower_is_better=lower_is_better,
            ),
        )[0]
        best_setup_by_comparison[key] = str(best["model_setup_id"])
        best_setup_rows.append(best)

    best_row_keys = {
        (model_type, function_name, sample_size, setup_id)
        for (model_type, function_name, sample_size), setup_id in best_setup_by_comparison.items()
    }
    best_rows = [row for row in final_rows if row.setup_key in best_row_keys]
    return (
        sorted(best_rows, key=sort_result_row),
        sorted(
            best_setup_rows,
            key=lambda row: (
                str(row["feynman_function"]),
                int(row["train_sample_size"]),
                str(row["model_type"]),
            ),
        ),
        best_setup_by_comparison,
    )


def aggregate_best_by_sample(
    best_rows: list[ResultRow],
    metric: str,
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    grouped = group_rows(best_rows, lambda row: (row.model_type, row.train_sample_size))
    for (model_type, sample_size), group in sorted(grouped.items()):
        stats = summarize_values(row.value(metric) for row in group)
        runtime_stats = summarize_values(row.runtime_seconds for row in group)
        table.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "train_sample_size": sample_size,
                "rows": len(group),
                "feynman_functions": len({row.feynman_function for row in group}),
                "random_seeds": len({row.random_seed for row in group}),
                f"{metric}_median": stats["median"],
                f"{metric}_q25": stats["q25"],
                f"{metric}_q75": stats["q75"],
                f"{metric}_mean": stats["mean"],
                f"{metric}_std": stats["std"],
                "runtime_seconds_median": runtime_stats["median"],
                "runtime_seconds_q25": runtime_stats["q25"],
                "runtime_seconds_q75": runtime_stats["q75"],
            }
        )
    return table


def aggregate_architecture_sweep(
    final_rows: list[ResultRow],
    metric: str,
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    grouped = group_rows(
        final_rows,
        lambda row: (row.model_type, row.model_setup_id, row.model_n_parameters),
    )
    for (model_type, setup_id, n_parameters), group in sorted(grouped.items()):
        stats = summarize_values(row.value(metric) for row in group)
        runtime_stats = summarize_values(row.runtime_seconds for row in group)
        table.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "model_setup_id": setup_id,
                "model_n_parameters": n_parameters,
                "rows": len(group),
                "feynman_functions": len({row.feynman_function for row in group}),
                "train_sample_sizes": len({row.train_sample_size for row in group}),
                "random_seeds": len({row.random_seed for row in group}),
                f"{metric}_median": stats["median"],
                f"{metric}_q25": stats["q25"],
                f"{metric}_q75": stats["q75"],
                f"{metric}_mean": stats["mean"],
                "runtime_seconds_median": runtime_stats["median"],
            }
        )
    return table


def win_rate_by_sample(
    best_rows: list[ResultRow],
    metric: str,
) -> list[dict[str, object]]:
    lower_is_better = metric_lower_is_better(metric)
    grouped = group_rows(
        best_rows,
        lambda row: (row.feynman_function, row.train_sample_size, row.random_seed),
    )
    wins: dict[tuple[int, str], int] = defaultdict(int)
    comparisons_by_sample: dict[int, int] = defaultdict(int)
    for (_, sample_size, _), group in grouped.items():
        if len({row.model_type for row in group}) < 2:
            continue
        winner = min(group, key=lambda row: row.value(metric))
        if not lower_is_better:
            winner = max(group, key=lambda row: row.value(metric))
        wins[(int(sample_size), winner.model_type)] += 1
        comparisons_by_sample[int(sample_size)] += 1

    model_types = sorted({row.model_type for row in best_rows})
    table: list[dict[str, object]] = []
    for sample_size in sorted(comparisons_by_sample):
        total = comparisons_by_sample[sample_size]
        for model_type in model_types:
            count = wins[(sample_size, model_type)]
            table.append(
                {
                    "train_sample_size": sample_size,
                    "model_type": model_type,
                    "label": model_label(model_type),
                    "wins": count,
                    "comparisons": total,
                    "win_share": count / total if total else float("nan"),
                }
            )
    return table


def family_gap_pairs(best_rows: list[ResultRow], metric: str) -> list[dict[str, object]]:
    lower_is_better = metric_lower_is_better(metric)
    grouped = group_rows(
        best_rows,
        lambda row: (
            row.feynman_function,
            row.train_sample_size,
            row.random_seed,
            row.model_family,
        ),
    )

    best_by_family: dict[tuple[str, int, int], dict[str, ResultRow]] = defaultdict(dict)
    for (function_name, sample_size, seed, family), group in grouped.items():
        chooser = min if lower_is_better else max
        best_by_family[(str(function_name), int(sample_size), int(seed))][str(family)] = chooser(
            group,
            key=lambda row: row.value(metric),
        )

    pairs: list[dict[str, object]] = []
    for (function_name, sample_size, seed), family_rows in best_by_family.items():
        if "kan" not in family_rows or "mlp" not in family_rows:
            continue
        kan_row = family_rows["kan"]
        mlp_row = family_rows["mlp"]
        kan_metric = kan_row.value(metric)
        mlp_metric = mlp_row.value(metric)
        gap = kan_metric - mlp_metric
        if lower_is_better:
            winner = "kan" if gap < 0 else "mlp"
        else:
            winner = "kan" if gap > 0 else "mlp"
        pairs.append(
            {
                "feynman_function": function_name,
                "train_sample_size": sample_size,
                "random_seed": seed,
                "kan_model_type": kan_row.model_type,
                "kan_setup_id": kan_row.model_setup_id,
                "kan_metric": kan_metric,
                "mlp_model_type": mlp_row.model_type,
                "mlp_setup_id": mlp_row.model_setup_id,
                "mlp_metric": mlp_metric,
                "kan_minus_mlp": gap,
                "winner_family": winner,
            }
        )
    return sorted(
        pairs,
        key=lambda row: (
            str(row["feynman_function"]),
            int(row["train_sample_size"]),
            int(row["random_seed"]),
        ),
    )


def family_gap_by_sample(
    pairs: list[dict[str, object]],
    *,
    metric: str,
) -> list[dict[str, object]]:
    lower_is_better = metric_lower_is_better(metric)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        grouped[int(row["train_sample_size"])].append(row)

    table: list[dict[str, object]] = []
    for sample_size, group in sorted(grouped.items()):
        gap_stats = summarize_values(float(row["kan_minus_mlp"]) for row in group)
        kan_wins = sum(1 for row in group if row["winner_family"] == "kan")
        table.append(
            {
                "train_sample_size": sample_size,
                "paired_comparisons": len(group),
                "kan_wins": kan_wins,
                "mlp_wins": len(group) - kan_wins,
                "kan_win_share": kan_wins / len(group) if group else float("nan"),
                "preferred_gap_direction": (
                    "negative_kan_better" if lower_is_better else "positive_kan_better"
                ),
                "kan_minus_mlp_median": gap_stats["median"],
                "kan_minus_mlp_q25": gap_stats["q25"],
                "kan_minus_mlp_q75": gap_stats["q75"],
                "kan_minus_mlp_mean": gap_stats["mean"],
            }
        )
    return table


def choose_dynamics_sample_size(rows: list[ResultRow], requested: int | None) -> int:
    available = sorted({row.train_sample_size for row in rows})
    if requested is None:
        return 100 if 100 in available else available[len(available) // 2]
    return min(available, key=lambda value: abs(value - requested))


def learning_curve_table(
    rows: list[ResultRow],
    best_setup_by_comparison: dict[tuple[str, str, int], str],
    metric: str,
    sample_size: int,
) -> list[dict[str, object]]:
    chosen_rows: list[ResultRow] = []
    for row in rows:
        setup_id = best_setup_by_comparison.get(
            (row.model_type, row.feynman_function, row.train_sample_size)
        )
        if setup_id is None:
            continue
        if row.train_sample_size == sample_size and row.model_setup_id == setup_id:
            chosen_rows.append(row)

    table: list[dict[str, object]] = []
    grouped = group_rows(chosen_rows, lambda row: (row.model_type, row.train_steps))
    for (model_type, train_steps), group in sorted(grouped.items()):
        stats = summarize_values(row.value(metric) for row in group)
        table.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "train_sample_size": sample_size,
                "train_steps": train_steps,
                "rows": len(group),
                f"{metric}_median": stats["median"],
                f"{metric}_q25": stats["q25"],
                f"{metric}_q75": stats["q75"],
                f"{metric}_mean": stats["mean"],
            }
        )
    return table


# ###########################################################################
# 5. Plots
# ###########################################################################


def setup_matplotlib(show: bool):
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    return plt


def should_log_y(metric: str, values: Iterable[float]) -> bool:
    if metric == "test_eval_r2":
        return False
    array = finite_array(values)
    return bool(array.size and np.all(array > 0.0))


def plot_sample_efficiency(plt, table: list[dict[str, object]], metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model_type in sorted({str(row["model_type"]) for row in table}):
        points = [row for row in table if row["model_type"] == model_type]
        points = sorted(points, key=lambda row: int(row["train_sample_size"]))
        x_values = np.asarray([int(row["train_sample_size"]) for row in points], dtype=float)
        medians = np.asarray([float(row[f"{metric}_median"]) for row in points], dtype=float)
        q25 = np.asarray([float(row[f"{metric}_q25"]) for row in points], dtype=float)
        q75 = np.asarray([float(row[f"{metric}_q75"]) for row in points], dtype=float)
        color = MODEL_COLORS.get(model_type)
        ax.plot(
            x_values,
            medians,
            marker=MODEL_MARKERS.get(model_type, "o"),
            label=model_label(model_type),
            color=color,
            linewidth=1.7,
        )
        ax.fill_between(x_values, q25, q75, color=color, alpha=0.13)

    ax.set_xscale("log")
    if should_log_y(metric, (float(row[f"{metric}_median"]) for row in table)):
        ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel(metric)
    ax.set_title("Final-checkpoint sample efficiency, best setup per model type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def plot_win_rate(plt, table: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for model_type in sorted({str(row["model_type"]) for row in table}):
        points = [row for row in table if row["model_type"] == model_type]
        points = sorted(points, key=lambda row: int(row["train_sample_size"]))
        x_values = np.asarray([int(row["train_sample_size"]) for row in points], dtype=float)
        y_values = np.asarray([float(row["win_share"]) for row in points], dtype=float)
        ax.plot(
            x_values,
            y_values,
            marker=MODEL_MARKERS.get(model_type, "o"),
            label=model_label(model_type),
            color=MODEL_COLORS.get(model_type),
            linewidth=1.7,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("training samples")
    ax.set_ylabel("win share")
    ax.set_title("Per-seed model-type wins after setup selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def plot_family_gap(
    plt,
    table: list[dict[str, object]],
    *,
    metric: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    points = sorted(table, key=lambda row: int(row["train_sample_size"]))
    x_values = np.asarray([int(row["train_sample_size"]) for row in points], dtype=float)
    medians = np.asarray([float(row["kan_minus_mlp_median"]) for row in points], dtype=float)
    q25 = np.asarray([float(row["kan_minus_mlp_q25"]) for row in points], dtype=float)
    q75 = np.asarray([float(row["kan_minus_mlp_q75"]) for row in points], dtype=float)
    ax.plot(x_values, medians, marker="o", color="#111827", linewidth=1.7)
    ax.fill_between(x_values, q25, q75, color="#64748b", alpha=0.2)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel(f"KAN minus MLP {metric}")
    direction = "below zero favors KAN" if metric_lower_is_better(metric) else "above zero favors KAN"
    ax.set_title(f"Best KAN family vs best MLP family ({direction})")
    fig.tight_layout()
    fig.savefig(path)


def plot_learning_curve(
    plt,
    table: list[dict[str, object]],
    metric: str,
    sample_size: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    for model_type in sorted({str(row["model_type"]) for row in table}):
        points = [row for row in table if row["model_type"] == model_type]
        points = sorted(points, key=lambda row: int(row["train_steps"]))
        x_values = np.asarray([int(row["train_steps"]) for row in points], dtype=float)
        medians = np.asarray([float(row[f"{metric}_median"]) for row in points], dtype=float)
        q25 = np.asarray([float(row[f"{metric}_q25"]) for row in points], dtype=float)
        q75 = np.asarray([float(row[f"{metric}_q75"]) for row in points], dtype=float)
        color = MODEL_COLORS.get(model_type)
        ax.plot(
            x_values,
            medians,
            marker=MODEL_MARKERS.get(model_type, "o"),
            label=model_label(model_type),
            color=color,
            linewidth=1.6,
        )
        ax.fill_between(x_values, q25, q75, color=color, alpha=0.13)
    if should_log_y(metric, (float(row[f"{metric}_median"]) for row in table)):
        ax.set_yscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning dynamics at {sample_size} training samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def plot_architecture_tradeoff(
    plt,
    table: list[dict[str, object]],
    metric: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for model_type in sorted({str(row["model_type"]) for row in table}):
        points = [row for row in table if row["model_type"] == model_type]
        x_values = np.asarray([int(row["model_n_parameters"]) for row in points], dtype=float)
        y_values = np.asarray([float(row[f"{metric}_median"]) for row in points], dtype=float)
        ax.scatter(
            x_values,
            y_values,
            label=model_label(model_type),
            color=MODEL_COLORS.get(model_type),
            marker=MODEL_MARKERS.get(model_type, "o"),
            alpha=0.78,
            s=52,
        )
    ax.set_xscale("log")
    if should_log_y(metric, (float(row[f"{metric}_median"]) for row in table)):
        ax.set_yscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel(metric)
    ax.set_title("Architecture sweep at final checkpoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


# ###########################################################################
# 6. Text summary
# ###########################################################################


def best_row_by_metric(
    rows: list[dict[str, object]],
    *,
    metric_column: str,
    lower_is_better: bool,
) -> dict[str, object] | None:
    if not rows:
        return None
    chooser = min if lower_is_better else max
    return chooser(rows, key=lambda row: float(row[metric_column]))


def summarize_overall_by_model_type(
    best_rows: list[ResultRow],
    metric: str,
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    grouped = group_rows(best_rows, lambda row: (row.model_type,))
    for (model_type,), group in sorted(grouped.items()):
        stats = summarize_values(row.value(metric) for row in group)
        runtime_stats = summarize_values(row.runtime_seconds for row in group)
        table.append(
            {
                "model_type": model_type,
                "label": model_label(str(model_type)),
                "rows": len(group),
                f"{metric}_median": stats["median"],
                f"{metric}_q25": stats["q25"],
                f"{metric}_q75": stats["q75"],
                "runtime_seconds_median": runtime_stats["median"],
            }
        )
    return table


def build_summary_text(
    *,
    load_report: LoadReport,
    rows: list[ResultRow],
    final_rows: list[ResultRow],
    final_step: int,
    metric: str,
    overall_table: list[dict[str, object]],
    sample_table: list[dict[str, object]],
    win_table: list[dict[str, object]],
    gap_pairs: list[dict[str, object]],
    gap_table: list[dict[str, object]],
    dynamics_sample_size: int,
) -> str:
    lower_is_better = metric_lower_is_better(metric)
    metric_column = f"{metric}_median"
    best_overall = best_row_by_metric(
        overall_table,
        metric_column=metric_column,
        lower_is_better=lower_is_better,
    )

    best_by_sample_lines: list[str] = []
    for sample_size in sorted({int(row["train_sample_size"]) for row in sample_table}):
        candidates = [row for row in sample_table if int(row["train_sample_size"]) == sample_size]
        best = best_row_by_metric(
            candidates,
            metric_column=metric_column,
            lower_is_better=lower_is_better,
        )
        if best is not None:
            best_by_sample_lines.append(
                f"  {sample_size:>4}: {best['label']} "
                f"({metric} median {float(best[metric_column]):.4g})"
            )

    gap_stats = summarize_values(float(row["kan_minus_mlp"]) for row in gap_pairs)
    if lower_is_better:
        kan_wins = sum(1 for row in gap_pairs if float(row["kan_minus_mlp"]) < 0.0)
    else:
        kan_wins = sum(1 for row in gap_pairs if float(row["kan_minus_mlp"]) > 0.0)
    kan_win_share = kan_wins / len(gap_pairs) if gap_pairs else float("nan")

    sample_win_lines: list[str] = []
    for row in sorted(gap_table, key=lambda item: int(item["train_sample_size"])):
        sample_win_lines.append(
            f"  {int(row['train_sample_size']):>4}: KAN family win share "
            f"{float(row['kan_win_share']):.1%}, median gap "
            f"{float(row['kan_minus_mlp_median']):.4g}"
        )

    lines = [
        "Experiment 1 result visualization summary",
        "",
        f"CSV files: {len(load_report.csv_paths)}",
        f"Raw rows: {load_report.raw_rows:,}",
        f"Rows after deduplication: {load_report.rows_after_deduplication:,}",
        f"Duplicate rows removed: {load_report.duplicate_rows:,}",
        f"Model types: {format_joined(row.model_type for row in rows)}",
        f"Feynman functions: {len({row.feynman_function for row in rows})}",
        f"Random seeds: {len({row.random_seed for row in rows})}",
        f"Training sample sizes: {format_joined(row.train_sample_size for row in rows)}",
        f"Final checkpoint used: {final_step}",
        f"Rows at final checkpoint: {len(final_rows):,}",
        "",
        "Overall best setup-selected model type:",
    ]
    if best_overall is not None:
        lines.append(
            f"  {best_overall['label']} with median {metric} "
            f"{float(best_overall[metric_column]):.4g}"
        )
    else:
        lines.append("  unavailable")

    lines.extend(
        [
            "",
            "Best model type by training-sample size:",
            *best_by_sample_lines,
            "",
            "Best KAN family vs best MLP family:",
            f"  paired comparisons: {len(gap_pairs):,}",
            f"  KAN family win share: {kan_win_share:.1%}",
            f"  median KAN-minus-MLP gap: {float(gap_stats['median']):.4g}",
            f"  q25/q75 KAN-minus-MLP gap: "
            f"{float(gap_stats['q25']):.4g} / {float(gap_stats['q75']):.4g}",
            *sample_win_lines,
            "",
            "Interpretation note:",
            "  Architecture selection is done on the final test metric within each",
            "  model type, Feynman function, and sample size. Treat these plots as",
            "  exploratory diagnostics, not a held-out model-selection estimate.",
            f"  Learning curves use the selected setup at {dynamics_sample_size} samples.",
        ]
    )
    if not win_table:
        lines.append("  Model-type win-rate table is empty because comparisons were missing.")
    return "\n".join(lines) + "\n"


# ###########################################################################
# 7. Main
# ###########################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing raw experiment CSV files.",
    )
    parser.add_argument(
        "--csv-glob",
        default="*.csv",
        help="Glob used inside --results-dir. Default: *.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures and summary CSVs.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_COLUMNS,
        default=DEFAULT_METRIC,
        help="Metric used for model selection and plots.",
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=None,
        help="Checkpoint step to use. Defaults to the largest train_steps value.",
    )
    parser.add_argument(
        "--dynamics-sample-size",
        type=int,
        default=100,
        help="Training-sample size for the learning-curve plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, load_report = load_all_results(args.results_dir, args.csv_glob)
    final_rows, final_step = final_checkpoint_rows(rows, args.final_step)
    best_rows, best_setup_table, best_setup_by_comparison = select_best_setups(
        final_rows,
        args.metric,
    )
    sample_table = aggregate_best_by_sample(best_rows, args.metric)
    architecture_table = aggregate_architecture_sweep(final_rows, args.metric)
    win_table = win_rate_by_sample(best_rows, args.metric)
    gap_pairs = family_gap_pairs(best_rows, args.metric)
    gap_table = family_gap_by_sample(gap_pairs, metric=args.metric)
    dynamics_sample_size = choose_dynamics_sample_size(rows, args.dynamics_sample_size)
    learning_table = learning_curve_table(
        rows,
        best_setup_by_comparison,
        args.metric,
        dynamics_sample_size,
    )
    overall_table = summarize_overall_by_model_type(best_rows, args.metric)

    write_csv_table(args.output_dir / "coverage_by_model_type.csv", coverage_table(rows))
    write_csv_table(args.output_dir / "overall_by_model_type.csv", overall_table)
    write_csv_table(args.output_dir / "best_setup_by_function_sample.csv", best_setup_table)
    write_csv_table(args.output_dir / "best_final_metric_by_sample.csv", sample_table)
    write_csv_table(args.output_dir / "architecture_sweep_summary.csv", architecture_table)
    write_csv_table(args.output_dir / "model_type_win_rate_by_sample.csv", win_table)
    write_csv_table(args.output_dir / "kan_vs_mlp_family_pairs.csv", gap_pairs)
    write_csv_table(args.output_dir / "kan_vs_mlp_family_gap_by_sample.csv", gap_table)
    write_csv_table(
        args.output_dir / f"learning_curve_sample_{dynamics_sample_size}.csv",
        learning_table,
    )

    summary_text = build_summary_text(
        load_report=load_report,
        rows=rows,
        final_rows=final_rows,
        final_step=final_step,
        metric=args.metric,
        overall_table=overall_table,
        sample_table=sample_table,
        win_table=win_table,
        gap_pairs=gap_pairs,
        gap_table=gap_table,
        dynamics_sample_size=dynamics_sample_size,
    )
    (args.output_dir / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")

    plt = setup_matplotlib(args.show)
    plot_sample_efficiency(
        plt,
        sample_table,
        args.metric,
        args.output_dir / "sample_efficiency.png",
    )
    if win_table:
        plot_win_rate(
            plt,
            win_table,
            args.output_dir / "model_type_win_rate.png",
        )
    if gap_table:
        plot_family_gap(
            plt,
            gap_table,
            metric=args.metric,
            path=args.output_dir / "kan_vs_mlp_family_gap.png",
        )
    if learning_table:
        plot_learning_curve(
            plt,
            learning_table,
            args.metric,
            dynamics_sample_size,
            args.output_dir / f"learning_curve_sample_{dynamics_sample_size}.png",
        )
    plot_architecture_tradeoff(
        plt,
        architecture_table,
        args.metric,
        args.output_dir / "architecture_tradeoff.png",
    )

    print(summary_text)
    print(f"Wrote visualizations and tables to {args.output_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
