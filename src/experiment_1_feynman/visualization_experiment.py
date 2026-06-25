"""Simple visualization script for experiment_simple.py results.

This file intentionally does not import torch.
Run it after experiment_simple.py has written results.csv.
"""


# %%
# ###########################################################################
# 1. Imports
# ###########################################################################

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# %%
# ###########################################################################
# 2. Parameters
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
LATEST_RESULTS_POINTER_PATH = THIS_DIR / "latest_result_path.txt"
if LATEST_RESULTS_POINTER_PATH.exists():
    RESULTS_CSV_PATH = Path(LATEST_RESULTS_POINTER_PATH.read_text(encoding="utf-8").strip())
else:
    RESULTS_CSV_PATH = THIS_DIR / "results.csv"
PARAMETER_SEPARATOR = "####### PARAMETERS #######"

METRIC_COLUMN = "test_eval_nrmse"
FEYNMAN_FUNCTION = "I.16.6"

TRAIN_STEPS_FOR_SAMPLE_PLOT = None
TRAIN_SAMPLE_SIZE_FOR_STEP_PLOT = 50

RUN_VISUALIZATION_NOW = True


# %%
# ###########################################################################
# 3. Load results.csv
# ###########################################################################


def load_results_csv(path: Path = RESULTS_CSV_PATH) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(PARAMETER_SEPARATOR):
            break
        data_lines.append(line)
    reader = csv.DictReader(data_lines)
    for row in reader:
        parsed = dict(row)
        for int_column in [
            "random_seed",
            "model_n_parameters",
            "train_sample_size",
            "train_steps",
        ]:
            parsed[int_column] = int(parsed[int_column])
        for float_column in [
            "train_loss",
            "test_loss",
            "runtime_seconds",
            "test_eval_mse",
            "test_eval_rmse",
            "test_eval_mae",
            "test_eval_nrmse",
            "test_eval_relative_l2",
            "test_eval_r2",
            "test_eval_max_absolute_error",
        ]:
            if float_column in parsed and parsed[float_column] not in {"", None}:
                parsed[float_column] = float(parsed[float_column])
        rows.append(parsed)
    return rows


if RESULTS_CSV_PATH.exists():
    results = load_results_csv(RESULTS_CSV_PATH)
else:
    results = []
    print(f"No results file found at {RESULTS_CSV_PATH}.")


# %%
# ###########################################################################
# 4. Helper functions
# ###########################################################################


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def latest_rows_by_seed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["random_seed"],
            row["feynman_function"],
            row["model_type"],
            row["train_sample_size"],
        )
        if key not in latest or row["train_steps"] > latest[key]["train_steps"]:
            latest[key] = row
    return list(latest.values())


def filter_rows(
    rows: list[dict[str, object]],
    *,
    feynman_function: str | None = None,
    train_steps: int | None = None,
    train_sample_size: int | None = None,
) -> list[dict[str, object]]:
    filtered = rows
    if feynman_function is not None:
        filtered = [row for row in filtered if row["feynman_function"] == feynman_function]
    if train_steps is not None:
        filtered = [row for row in filtered if row["train_steps"] == train_steps]
    if train_sample_size is not None:
        filtered = [
            row for row in filtered if row["train_sample_size"] == train_sample_size
        ]
    return filtered


def grouped_mean_std(
    rows: list[dict[str, object]],
    *,
    x_column: str,
    y_column: str,
) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_type"], row[x_column])].append(row[y_column])

    output: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (model_type, x_value), values in grouped.items():
        mean, std = mean_std(values)
        output[model_type].append((int(x_value), mean, std))

    for model_type in output:
        output[model_type] = sorted(output[model_type], key=lambda item: item[0])
    return output


# %%
# ###########################################################################
# 5. Visualize metric over training sample sizes
# ###########################################################################


def plot_metric_over_train_sample_size(
    rows: list[dict[str, object]],
    *,
    metric_column: str = METRIC_COLUMN,
    feynman_function: str = FEYNMAN_FUNCTION,
    train_steps: int | None = TRAIN_STEPS_FOR_SAMPLE_PLOT,
) -> None:
    filtered = filter_rows(rows, feynman_function=feynman_function)
    if train_steps is None:
        filtered = latest_rows_by_seed(filtered)
        title_steps = "latest checkpoint"
    else:
        filtered = filter_rows(filtered, train_steps=train_steps)
        title_steps = f"{train_steps} train steps"

    grouped = grouped_mean_std(
        filtered,
        x_column="train_sample_size",
        y_column=metric_column,
    )

    plt.figure(figsize=(7.0, 4.5))
    for model_type, points in grouped.items():
        x_values = np.array([point[0] for point in points], dtype=np.float64)
        means = np.array([point[1] for point in points], dtype=np.float64)
        stds = np.array([point[2] for point in points], dtype=np.float64)
        plt.plot(x_values, means, marker="o", label=model_type)
        plt.fill_between(x_values, means - stds, means + stds, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("training samples")
    plt.ylabel(metric_column)
    plt.title(f"{metric_column} on {feynman_function} at {title_steps}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()


# %%
# ###########################################################################
# 6. Visualize metric over training steps
# ###########################################################################


def plot_metric_over_train_steps(
    rows: list[dict[str, object]],
    *,
    metric_column: str = METRIC_COLUMN,
    feynman_function: str = FEYNMAN_FUNCTION,
    train_sample_size: int = TRAIN_SAMPLE_SIZE_FOR_STEP_PLOT,
) -> None:
    filtered = filter_rows(
        rows,
        feynman_function=feynman_function,
        train_sample_size=train_sample_size,
    )
    grouped = grouped_mean_std(
        filtered,
        x_column="train_steps",
        y_column=metric_column,
    )

    plt.figure(figsize=(7.0, 4.5))
    for model_type, points in grouped.items():
        x_values = np.array([point[0] for point in points], dtype=np.float64)
        means = np.array([point[1] for point in points], dtype=np.float64)
        stds = np.array([point[2] for point in points], dtype=np.float64)
        plt.plot(x_values, means, marker="o", label=model_type)
        plt.fill_between(x_values, means - stds, means + stds, alpha=0.15)

    plt.xlabel("training steps")
    plt.ylabel(metric_column)
    plt.title(
        f"{metric_column} on {feynman_function} "
        f"with {train_sample_size} training samples"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()


if RUN_VISUALIZATION_NOW and results:
    plot_metric_over_train_sample_size(results)
    plot_metric_over_train_steps(results)
    plt.show()
