from pathlib import Path
import csv
from collections import defaultdict

import numpy as np


HERE = Path(__file__).resolve().parent
SMOOTHED_DIR = HERE / "smoothed_csv"
PARAMETER_SEPARATOR = "####### PARAMETERS #######"
METRIC = "test_eval_nrmse"
FEYNMAN_FUNCTION = None
TARGET_PARAMS = 180
TRAIN_STEPS = 1500

PANELS = [
    ("kan_bspline", "KAN (B-Spline)", "#2563eb"),
    ("kan_gaussrbf", "KAN (RBF)", "#0f766e"),
    ("mlp_sigmoid", "MLP (Sigmoid)", "#9333ea"),
    ("mlp_relu", "MLP (ReLU)", "#dc2626"),
]


def csv_data_lines(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith(PARAMETER_SEPARATOR):
                break
            yield line


def load_rows(metric=METRIC, feynman_function=FEYNMAN_FUNCTION, train_steps=None):
    rows = []
    csv_files = sorted(SMOOTHED_DIR.glob("*_sm.csv"))

    for path in csv_files:
        for row in csv.DictReader(csv_data_lines(path)):
            if feynman_function and row["feynman_function"] != feynman_function:
                continue
            if train_steps is not None and int(row["train_steps"]) != train_steps:
                continue
            if not row.get(metric):
                continue
            rows.append(
                {
                    "seed": int(row["random_seed"]),
                    "func": row["feynman_function"],
                    "model": row["model_type"],
                    "setup": row.get("model_setup_id", row["model_dims"]),
                    "dims": row["model_dims"],
                    "params": int(row["model_n_parameters"]),
                    "samples": int(row["train_sample_size"]),
                    "steps": int(row["train_steps"]),
                    "nrmse": float(row[metric]),
                }
            )
    return rows


def seed_mean_std(values):
    by_seed = defaultdict(list)
    for seed, value in values:
        by_seed[seed].append(value)
    seed_values = [np.mean(seed_values) for seed_values in by_seed.values()]
    if len(seed_values) == 0:
        return np.nan, np.nan
    if len(seed_values) == 1:
        return float(seed_values[0]), 0.0
    return float(np.mean(seed_values)), float(np.std(seed_values, ddof=1))


def mean_std_by_sample(rows):
    samples = sorted({row["samples"] for row in rows})
    means, stds = [], []
    for sample in samples:
        values = [(row["seed"], row["nrmse"]) for row in rows if row["samples"] == sample]
        mean, std = seed_mean_std(values)
        means.append(mean)
        stds.append(std)
    return np.array(samples), np.array(means), np.array(stds)


def best_over_steps_by_setup(rows):
    best = {}
    for row in rows:
        key = (row["seed"], row["func"], row["model"], row["dims"], row["params"], row["samples"])
        if key not in best or row["nrmse"] < best[key]["nrmse"]:
            best[key] = row
    return list(best.values())


def best_over_architecture_at_each_step(rows):
    best = {}
    for row in rows:
        key = (row["seed"], row["func"], row["model"], row["samples"], row["steps"])
        if key not in best or row["nrmse"] < best[key]["nrmse"]:
            best[key] = row
    return list(best.values())


def select_closest_parameter_best_setup(rows, target_params=TARGET_PARAMS):
    selected = {}
    for model, _, _ in PANELS:
        model_rows = [row for row in rows if row["model"] == model]
        params = sorted({row["params"] for row in model_rows})
        chosen_params = min(params, key=lambda value: (abs(value - target_params), value))
        candidates = [row for row in model_rows if row["params"] == chosen_params]
        setup_scores = defaultdict(list)
        for row in candidates:
            setup_scores[(row["setup"], row["dims"], row["params"])].append(row["nrmse"])
        selected[model] = min(setup_scores, key=lambda key: np.mean(setup_scores[key]))
    return selected


def filter_selected_rows(rows, selected):
    return [
        row
        for row in rows
        if row["model"] in selected
        and (row["setup"], row["dims"], row["params"]) == selected[row["model"]]
    ]


def best_family_values(selected_rows):
    best = {}
    for row in selected_rows:
        family = "kan" if row["model"].startswith("kan") else "mlp"
        key = (row["seed"], row["func"], row["samples"], family)
        if key not in best or row["nrmse"] < best[key]:
            best[key] = row["nrmse"]
    return best

