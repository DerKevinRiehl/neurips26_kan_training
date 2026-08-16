from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from experiment1_common_sm import (
    HERE,
    PANELS,
    TARGET_PARAMS,
    TRAIN_STEPS,
    best_over_architecture_at_each_step,
    best_over_steps_by_setup,
    filter_selected_rows,
    load_rows,
    mean_std_by_sample,
    seed_mean_std,
    select_closest_parameter_best_setup,
)


FIG_RES = 5.5
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
fig = plt.figure(figsize=(FIG_RES * 2.12, FIG_RES * 1.55), constrained_layout=True)

rows_all = load_rows()
rows_fixed_steps = load_rows(train_steps=TRAIN_STEPS)


best_complexity = best_over_steps_by_setup(rows_all)
complexity = defaultdict(list)
for row in best_complexity:
    complexity[(row["model"], row["params"], row["samples"])].append((row["seed"], row["nrmse"]))

best_iterations = best_over_architecture_at_each_step(rows_all)
iterations = defaultdict(list)
for row in best_iterations:
    iterations[(row["model"], row["steps"], row["samples"])].append((row["seed"], row["nrmse"]))

selected = select_closest_parameter_best_setup(rows_fixed_steps, TARGET_PARAMS)
selected_rows = filter_selected_rows(rows_fixed_steps, selected)


all_params = sorted({key[1] for key in complexity})
param_cmap = plt.cm.viridis
param_norm = Normalize(vmin=min(all_params), vmax=max(all_params))
row1_axes = []

for panel_id, (model, title, _) in enumerate(PANELS, start=1):
    ax = plt.subplot(3, 4, panel_id)
    row1_axes.append(ax)
    params = sorted({key[1] for key in complexity if key[0] == model})
    for n_params in params:
        samples = sorted({key[2] for key in complexity if key[0] == model and key[1] == n_params})
        means = [seed_mean_std(complexity[(model, n_params, sample)])[0] for sample in samples]
        plt.plot(
            samples,
            means,
            marker="o",
            markersize=2.7,
            linewidth=1.1,
            color=param_cmap(param_norm(n_params)),
        )
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.25)
    if panel_id == 1:
        plt.ylabel("(a) model complexity\n\nNRMSE", fontweight="bold")
    else:
        plt.yticks([])

param_sm = ScalarMappable(norm=param_norm, cmap=param_cmap)
param_sm.set_array([])
param_cbar = fig.colorbar(param_sm, ax=row1_axes, pad=0.018, fraction=0.035, aspect=16)
param_cbar.set_label("# parameters", rotation=270, labelpad=11)


all_steps = sorted({key[1] for key in iterations})
step_cmap = plt.cm.plasma
step_norm = Normalize(vmin=min(all_steps), vmax=max(all_steps))
row2_axes = []

for panel_id, (model, title, _) in enumerate(PANELS, start=5):
    ax = plt.subplot(3, 4, panel_id)
    row2_axes.append(ax)
    steps = sorted({key[1] for key in iterations if key[0] == model})
    for train_steps in steps:
        samples = sorted({key[2] for key in iterations if key[0] == model and key[1] == train_steps})
        means = [seed_mean_std(iterations[(model, train_steps, sample)])[0] for sample in samples]
        plt.plot(
            samples,
            means,
            marker="o",
            markersize=2.7,
            linewidth=1.1,
            color=step_cmap(step_norm(train_steps)),
        )
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    if panel_id == 5:
        plt.ylabel("(b) training iterations\n\nNRMSE", fontweight="bold")
    else:
        plt.yticks([])

step_sm = ScalarMappable(norm=step_norm, cmap=step_cmap)
step_sm.set_array([])
step_cbar = fig.colorbar(step_sm, ax=row2_axes, pad=0.018, fraction=0.035, aspect=16)
step_cbar.set_label("# training steps", rotation=270, labelpad=13)


for panel_id, (model, title, color) in enumerate(PANELS, start=9):
    plt.subplot(3, 4, panel_id)
    data = [row for row in selected_rows if row["model"] == model]
    samples, means, stds = mean_std_by_sample(data)
    params = selected[model][2]
    dims = selected[model][1]
    plt.plot(samples, means, marker="o", markersize=3.0, linewidth=1.3, color=color, label=f"{params} params, {dims}")
    plt.fill_between(samples, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("# Samples")
    plt.grid(True, alpha=0.25)
    if panel_id == 9:
        plt.ylabel("(c) best-model sample efficiency\n\nNRMSE", fontweight="bold")
        # plt.legend(fontsize=5.7, frameon=False)
    else:
        plt.yticks([])

plt.show()
