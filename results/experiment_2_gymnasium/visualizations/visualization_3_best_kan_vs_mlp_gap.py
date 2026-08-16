import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiment2_common import (
    FIGURE_DIR,
    MODEL_COLORS,
    best_setup_per_group,
    find_result_csv,
    load_results,
    mean_std_curve,
    safe_name,
)


FIGURE_DIR.mkdir(exist_ok=True)

csv_path = find_result_csv()
df = load_results(csv_path)
best = best_setup_per_group(df, ["environment_id", "model_family"])
gap_rows = []

for env in sorted(df["environment_id"].unique()):
    kan_row = best[(best["environment_id"] == env) & (best["model_family"] == "kan")].iloc[0]
    mlp_row = best[(best["environment_id"] == env) & (best["model_family"] == "mlp")].iloc[0]

    kan_curve = mean_std_curve(df, env, kan_row["model_setup_id"])
    mlp_curve = mean_std_curve(df, env, mlp_row["model_setup_id"])

    kan_seed = df[(df["environment_id"] == env) & (df["model_setup_id"] == kan_row["model_setup_id"])][
        ["random_seed", "train_episodes", "eval_return_mean", "train_env_steps_total"]
    ].rename(columns={"eval_return_mean": "kan_return", "train_env_steps_total": "kan_steps"})
    mlp_seed = df[(df["environment_id"] == env) & (df["model_setup_id"] == mlp_row["model_setup_id"])][
        ["random_seed", "train_episodes", "eval_return_mean", "train_env_steps_total"]
    ].rename(columns={"eval_return_mean": "mlp_return", "train_env_steps_total": "mlp_steps"})

    merged = kan_seed.merge(mlp_seed, on=["random_seed", "train_episodes"], how="inner")
    merged["gap"] = merged["kan_return"] - merged["mlp_return"]
    merged["steps_mean"] = 0.5 * (merged["kan_steps"] + merged["mlp_steps"])
    gap = (
        merged.groupby("train_episodes", as_index=False)
        .agg(gap_mean=("gap", "mean"), gap_std=("gap", "std"), steps_mean=("steps_mean", "mean"))
        .sort_values("train_episodes")
    )
    gap["environment_id"] = env
    gap["best_kan_setup_id"] = kan_row["model_setup_id"]
    gap["best_mlp_setup_id"] = mlp_row["model_setup_id"]
    gap_rows.append(gap)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.2),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    for curve, row, family, label in [
        (kan_curve, kan_row, "kan", f"Best KAN: {kan_row['model_setup_id']}"),
        (mlp_curve, mlp_row, "mlp", f"Best MLP: {mlp_row['model_setup_id']}"),
    ]:
        x = curve["train_env_steps_mean"].to_numpy() / 1000.0
        y = curve["eval_return_mean"].to_numpy()
        y_std = curve["eval_return_std"].fillna(0.0).to_numpy()
        axes[0].plot(x, y, color=MODEL_COLORS[family], linewidth=2.0, label=label)
        axes[0].fill_between(x, y - y_std, y + y_std, color=MODEL_COLORS[family], alpha=0.15, linewidth=0)

    axes[0].set_title(f"{env}: best KAN family versus best MLP family")
    axes[0].set_xlabel("Environment steps during training (thousands)")
    axes[0].set_ylabel("Evaluation return")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    x_gap = gap["steps_mean"].to_numpy() / 1000.0
    y_gap = gap["gap_mean"].to_numpy()
    y_gap_std = gap["gap_std"].fillna(0.0).to_numpy()
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].plot(x_gap, y_gap, color="#333333", linewidth=1.8)
    axes[1].fill_between(x_gap, y_gap - y_gap_std, y_gap + y_gap_std, color="#333333", alpha=0.15, linewidth=0)
    axes[1].set_xlabel("Environment steps during training (thousands)")
    axes[1].set_ylabel("KAN - MLP return")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = FIGURE_DIR / f"{safe_name(env)}__best_kan_vs_mlp_gap.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(out_path)

pd.concat(gap_rows, ignore_index=True).to_csv(FIGURE_DIR / "best_kan_vs_mlp_gap_curves.csv", index=False)
