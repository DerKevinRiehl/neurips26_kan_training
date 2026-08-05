from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import os


METRIC = "test_eval_nrmse"
FEYNMAN_FUNCTION = None  # Example: "I.16.6"; None means average functions within each seed.
TARGET_PARAMS = 180
TRAIN_STEPS = 1500
PARAMETER_SEPARATOR = "####### PARAMETERS #######"
PANELS = [
    ("kan_bspline", "KAN B"),
    ("kan_gaussrbf", "KAN RBF"),
    ("mlp_sigmoid", "MLP SIG"),
    ("mlp_relu", "MLP RELU"),
]

CSV_FILES = sorted(os.listdir(".."))
CSV_FILES = ["../" + file for file in CSV_FILES if file.endswith(".csv")]

rows = []
for path in [Path(path) for path in CSV_FILES]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(PARAMETER_SEPARATOR):
            break
        lines.append(line)
    for row in csv.DictReader(lines):
        if FEYNMAN_FUNCTION and row["feynman_function"] != FEYNMAN_FUNCTION:
            continue
        if int(row["train_steps"]) != TRAIN_STEPS or not row.get(METRIC):
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
                "nrmse": float(row[METRIC]),
            }
        )


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


selected = {}
for model, _ in PANELS:
    model_rows = [row for row in rows if row["model"] == model]
    params = sorted({row["params"] for row in model_rows})
    chosen_params = min(params, key=lambda value: (abs(value - TARGET_PARAMS), value))
    candidates = [row for row in model_rows if row["params"] == chosen_params]
    setup_scores = defaultdict(list)
    for row in candidates:
        setup_scores[(row["setup"], row["dims"], row["params"])].append(row["nrmse"])
    selected[model] = min(setup_scores, key=lambda key: np.mean(setup_scores[key]))

selected_rows = [
    row
    for row in rows
    if row["model"] in selected
    and (row["setup"], row["dims"], row["params"]) == selected[row["model"]]
]

plt.figure(figsize=(12, 11))
for panel_id, (model, title) in enumerate(PANELS, start=1):
    plt.subplot(3, 2, panel_id)
    data = [row for row in selected_rows if row["model"] == model]
    samples = sorted({row["samples"] for row in data})
    means, stds = [], []
    for sample in samples:
        values = [(row["seed"], row["nrmse"]) for row in data if row["samples"] == sample]
        mean, std = seed_mean_std(values)
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    params = selected[model][2]
    dims = selected[model][1]
    plt.plot(samples, means, marker="o", label=f"{params} params")
    plt.fill_between(samples, means - stds, means + stds, alpha=0.18)
    plt.xscale("log")
    plt.xlabel("samples")
    plt.ylabel("NRMSE")
    plt.ylim(0.0, 1.0)
    plt.title(f"{title}, {params} params, {dims}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)

best_by_family = {}
for row in selected_rows:
    family = "kan" if row["model"].startswith("kan") else "mlp"
    key = (row["seed"], row["func"], row["samples"], family)
    if key not in best_by_family or row["nrmse"] < best_by_family[key]:
        best_by_family[key] = row["nrmse"]

diffs = defaultdict(list)
for seed, func, sample, family in list(best_by_family):
    if family != "kan":
        continue
    kan = best_by_family.get((seed, func, sample, "kan"))
    mlp = best_by_family.get((seed, func, sample, "mlp"))
    if kan is not None and mlp is not None:
        diffs[sample].append((seed, kan - mlp))

plt.subplot(3, 1, 3)
samples = sorted(diffs)
means, stds = [], []
for sample in samples:
    mean, std = seed_mean_std(diffs[sample])
    means.append(mean)
    stds.append(std)
means = np.array(means)
stds = np.array(stds)
plt.plot(samples, means, marker="o", color="black", label="best KAN - best MLP")
plt.fill_between(samples, means - stds, means + stds, color="black", alpha=0.15)
plt.axhline(0.0, color="gray", linewidth=1)
plt.xscale("log")
plt.xlabel("samples")
plt.ylabel("NRMSE difference")
plt.title("Best KAN NRMSE - best MLP NRMSE")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)

plt.tight_layout()
plt.show()
