from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from experiment1_common import (
    HERE,
    TARGET_PARAMS,
    TRAIN_STEPS,
    best_family_values,
    filter_selected_rows,
    load_rows,
    seed_mean_std,
    select_closest_parameter_best_setup,
)


FIG_RES = 5.5
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
plt.figure(figsize=(FIG_RES * 2, FIG_RES * 0.50), constrained_layout=True)

rows = load_rows(train_steps=TRAIN_STEPS)
selected = select_closest_parameter_best_setup(rows, TARGET_PARAMS)
selected_rows = filter_selected_rows(rows, selected)
best_by_family = best_family_values(selected_rows)

family_values = defaultdict(list)
relative_improvements = defaultdict(list)


def samples_to_reach_nrmse(sample_values, nrmse_values, target_nrmse):
    sample_values = np.asarray(sample_values, dtype=float)
    nrmse_values = np.asarray(nrmse_values, dtype=float)
    order = np.argsort(sample_values)
    sample_values = sample_values[order]
    nrmse_values = nrmse_values[order]
    nrmse_envelope = np.minimum.accumulate(nrmse_values)

    if target_nrmse >= nrmse_envelope[0]:
        return sample_values[0]
    if target_nrmse < nrmse_envelope[-1]:
        return np.nan

    idx = np.where(nrmse_envelope <= target_nrmse)[0][0]
    if idx == 0:
        return sample_values[0]

    x0 = np.log10(sample_values[idx - 1])
    x1 = np.log10(sample_values[idx])
    y0 = nrmse_envelope[idx - 1]
    y1 = nrmse_envelope[idx]

    if abs(y1 - y0) < 1e-12:
        return sample_values[idx]

    weight = (target_nrmse - y0) / (y1 - y0)
    return 10.0 ** (x0 + weight * (x1 - x0))


for seed, func, sample, family in list(best_by_family):
    if family != "kan":
        continue
    kan = best_by_family.get((seed, func, sample, "kan"))
    mlp = best_by_family.get((seed, func, sample, "mlp"))
    if kan is None or mlp is None:
        continue
    family_values[("kan", sample)].append((seed, kan))
    family_values[("mlp", sample)].append((seed, mlp))
    relative_improvements[sample].append((seed, 100.0 * (mlp - kan) / max(mlp, 1e-12)))


family_summary = {}
for family in ["kan", "mlp"]:
    samples = sorted({key[1] for key in family_values if key[0] == family})
    means, stds = [], []
    for sample in samples:
        mean, std = seed_mean_std(family_values[(family, sample)])
        means.append(mean)
        stds.append(std)
    family_summary[family] = (np.array(samples), np.array(means), np.array(stds))

samples = sorted(relative_improvements)
means, stds = [], []
for sample in samples:
    mean, std = seed_mean_std(relative_improvements[sample])
    means.append(mean)
    stds.append(std)
improvement_summary = (np.array(samples), np.array(means), np.array(stds))


plt.subplot(1, 4, 1)
for family, label, color in [
    ("kan", "KAN (best)", "#2563eb"),
    ("mlp", "MLP (best)", "#dc2626"),
]:
    samples, means, stds = family_summary[family]
    plt.plot(samples, means, marker="o", linewidth=1.8, color=color, label=label)
    plt.fill_between(samples, means - stds, means + stds, color=color, alpha=0.14, linewidth=0)

plt.xscale("log")
plt.ylim(0.0, 1.0)
plt.xlabel("# Samples")
plt.ylabel("NRMSE")
# plt.title("(a) absolute error", fontweight="bold")
plt.grid(True, alpha=0.25)
plt.legend(frameon=False)


plt.subplot(1, 4, 2)
samples, means, stds = improvement_summary

plt.plot(samples, means, marker="o", linewidth=1.8, color="black", label="Best KAN vs. best MLP")
plt.fill_between(samples, means - stds, means + stds, color="black", alpha=0.14, linewidth=0)
plt.axhline(0.0, color="gray", linewidth=1.0)
plt.xscale("log")
plt.xlabel("# Samples")
plt.ylabel("Relative NRMSE improvement (%)")
# plt.title("(b) relative improvement", fontweight="bold")
plt.grid(True, alpha=0.25)


plt.subplot(1, 4, 3)
mlp_samples, mlp_means, mlp_stds = family_summary["mlp"]
improvement_samples, improvement_means, improvement_stds = improvement_summary
mlp_lookup = {sample: (mean, std) for sample, mean, std in zip(mlp_samples, mlp_means, mlp_stds)}

x, xerr, y, yerr = [], [], [], []
for sample, mean, std in zip(improvement_samples, improvement_means, improvement_stds):
    if sample not in mlp_lookup:
        continue
    mlp_mean, mlp_std = mlp_lookup[sample]
    x.append(mlp_mean)
    xerr.append(mlp_std)
    y.append(mean)
    yerr.append(std)

plt.errorbar(x, y, xerr=xerr, yerr=yerr, marker="o", linewidth=1.8, color="black", capsize=2)
plt.axhline(0.0, color="gray", linewidth=1.0)
plt.xlabel("Best MLP NRMSE")
plt.ylabel("Relative NRMSE improvement (%)")
# plt.title("(c) improvement by error level", fontweight="bold")
plt.grid(True, alpha=0.25)
plt.gca().invert_xaxis()


plt.subplot(1, 4, 4)
kan_samples, kan_means, _ = family_summary["kan"]
mlp_samples, mlp_means, _ = family_summary["mlp"]

x, y = [], []
for mlp_sample, mlp_nrmse in zip(mlp_samples, mlp_means):
    kan_sample = samples_to_reach_nrmse(kan_samples, kan_means, mlp_nrmse)
    if np.isnan(kan_sample):
        continue
    x.append(mlp_nrmse)
    y.append(100.0 * (mlp_sample - kan_sample) / mlp_sample)

plt.plot(x, y, marker="o", linewidth=1.8, color="black")
plt.axhline(0.0, color="gray", linewidth=1.0)
plt.xlabel("Best MLP NRMSE")
plt.ylabel("Sample reduction (%)")
# plt.title("(d) sample reduction", fontweight="bold")
plt.grid(True, alpha=0.25)
plt.gca().invert_xaxis()

plt.show()
