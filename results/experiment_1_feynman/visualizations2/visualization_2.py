from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import os

METRIC = "test_eval_nrmse"
FEYNMAN_FUNCTION = None  # Example: "I.16.6"; None means all functions.
CSV_FILES = []  # Example: [r"C:\path\to\20260626_1704_feynman_protocol_result.csv"]
PARAMETER_SEPARATOR = "####### PARAMETERS #######"
PANELS = [
    ("kan_bspline", "KAN B"),
    ("kan_gaussrbf", "KAN RBF"),
    ("mlp_sigmoid", "MLP SIG"),
    ("mlp_relu", "MLP RELU"),
]

CSV_FILES = sorted(os.listdir(".."))
CSV_FILES = ["../"+file for file in CSV_FILES if file.endswith(".csv")]

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
        if not row.get(METRIC):
            continue
        rows.append(
            {
                "seed": int(row["random_seed"]),
                "func": row["feynman_function"],
                "model": row["model_type"],
                "dims": row["model_dims"],
                "params": int(row["model_n_parameters"]),
                "samples": int(row["train_sample_size"]),
                "steps": int(row["train_steps"]),
                "nrmse": float(row[METRIC]),
            }
        )

best = {}
for row in rows:
    key = (row["seed"], row["func"], row["model"], row["dims"], row["params"], row["samples"])
    if key not in best or row["nrmse"] < best[key]["nrmse"]:
        best[key] = row

grouped = defaultdict(list)
for row in best.values():
    grouped[(row["model"], row["steps"], row["samples"])].append(row["nrmse"])

plt.figure(figsize=(12, 8))
for panel_id, (model, title) in enumerate(PANELS, start=1):
    plt.subplot(2, 2, panel_id)
    steps = sorted({key[1] for key in grouped if key[0] == model})
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(steps), 1)))
    for color, train_steps in zip(colors, steps):
        samples = sorted({key[2] for key in grouped if key[0] == model and key[1] == train_steps})
        y = [np.mean(grouped[(model, train_steps, sample)]) for sample in samples]
        plt.plot(samples, y, marker="o", color=color, label=f"{train_steps} steps")
    plt.xscale("log")
    plt.xlabel("samples")
    plt.ylabel("NRMSE")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)

plt.tight_layout()
plt.show()
