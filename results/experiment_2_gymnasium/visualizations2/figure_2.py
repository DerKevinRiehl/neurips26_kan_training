from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIG_RES = 5.5
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.figure(figsize=(FIG_RES * 2, FIG_RES), constrained_layout=True)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent
CSV_FILE = max(RESULTS_DIR.glob("*_experiment_classic_result.csv"), key=lambda path: path.stat().st_size)

USECOLS = [
    "random_seed",
    "environment_id",
    "model_setup_id",
    "model_type",
    "total_n_parameters",
    "train_episodes",
    "train_env_steps_total",
    "eval_return_mean",
]

FAMILIES = [
    ("kan", "KAN (best)", "#2563eb"),
    ("mlp", "MLP (best)", "#dc2626"),
]

ENV_ORDER = [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
]

ENV_LABELS = {
    "Acrobot-v1": "Acrobot",
    "CartPole-v1": "CartPole",
    "MountainCarContinuous-v0": "Mountain Car Continuous",
    "MountainCar-v0": "Mountain Car",
    "Pendulum-v1": "Pendulum",
}


def add_training_step_axis(reference_curves):
    if not reference_curves:
        return

    reference = (
        pd.concat(reference_curves)
        .groupby("train_episodes")["train_env_steps_total"]
        .mean()
        .sort_index()
    )
    episode_values = reference.index.to_numpy(dtype=float)

    if len(episode_values) <= 5:
        ticks = episode_values
    else:
        tick_ids = np.linspace(0, len(episode_values) - 1, 5).round().astype(int)
        ticks = episode_values[np.unique(tick_ids)]

    step_labels = []
    for tick in ticks:
        steps = reference.loc[int(tick)] / 1000.0
        step_labels.append(f"{steps:.0f}" if steps >= 10.0 else f"{steps:.1f}")

    ax = plt.gca()
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(ticks)
    top_ax.set_xticklabels(step_labels)
    top_ax.set_xlabel("# Training steps (k)")
    top_ax.tick_params(axis="x", labelsize=7)


df = pd.read_csv(CSV_FILE, usecols=USECOLS, low_memory=False)
df = df[df["environment_id"].astype(str).str.contains("-v[0-9]", regex=True, na=False)].copy()
df = df[df["model_type"].astype(str).str.startswith(("kan", "mlp"))].copy()

for col in ["random_seed", "total_n_parameters", "train_episodes", "train_env_steps_total", "eval_return_mean"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["random_seed", "train_episodes", "train_env_steps_total", "eval_return_mean"])
df = df[df["random_seed"].between(0, 999)].copy()
df["random_seed"] = df["random_seed"].astype(int)
df["train_episodes"] = df["train_episodes"].astype(int)
df["model_family"] = np.where(df["model_type"].astype(str).str.startswith("kan"), "kan", "mlp")

final_checkpoint = int(df["train_episodes"].max())
final_by_seed = (
    df[df["train_episodes"] == final_checkpoint]
    .groupby(["environment_id", "model_family", "model_type", "model_setup_id", "random_seed"], as_index=False)
    .agg(
        eval_return_mean=("eval_return_mean", "mean"),
        total_n_parameters=("total_n_parameters", "first"),
    )
)

final_by_setup = (
    final_by_seed.groupby(["environment_id", "model_family", "model_type", "model_setup_id"], as_index=False)
    .agg(
        eval_return_mean=("eval_return_mean", "mean"),
        total_n_parameters=("total_n_parameters", "first"),
        seeds=("random_seed", "nunique"),
    )
)

best_setup = final_by_setup.loc[
    final_by_setup.groupby(["environment_id", "model_family"])["eval_return_mean"].idxmax()
].copy()

available_envs = set(df["environment_id"].unique())
envs = [env for env in ENV_ORDER if env in available_envs]
envs += sorted(available_envs.difference(envs))
envs = envs[:5]

for panel_id, env in enumerate(envs, start=1):
    plt.subplot(2, 3, panel_id)
    axis_reference = []

    for family, label, color in FAMILIES:
        row = best_setup[(best_setup["environment_id"] == env) & (best_setup["model_family"] == family)]
        if row.empty:
            continue

        setup_id = row.iloc[0]["model_setup_id"]
        model_type = row.iloc[0]["model_type"]
        sub = df[
            (df["environment_id"] == env)
            & (df["model_family"] == family)
            & (df["model_type"] == model_type)
            & (df["model_setup_id"] == setup_id)
        ].copy()

        seed_curve = (
            sub.groupby(["train_episodes", "random_seed"], as_index=False)
            .agg(
                train_env_steps_total=("train_env_steps_total", "mean"),
                eval_return_mean=("eval_return_mean", "mean"),
            )
            .sort_values("train_episodes")
        )

        curve = (
            seed_curve.groupby("train_episodes", as_index=False)
            .agg(
                train_env_steps_total=("train_env_steps_total", "mean"),
                eval_return_mean=("eval_return_mean", "mean"),
                eval_return_std=("eval_return_mean", "std"),
            )
            .sort_values("train_episodes")
        )

        axis_reference.append(curve[["train_episodes", "train_env_steps_total"]])
        x = curve["train_episodes"].to_numpy()
        y = curve["eval_return_mean"].to_numpy()
        y_std = curve["eval_return_std"].fillna(0.0).to_numpy()

        plt.plot(x, y, color=color, linewidth=1.8)
        plt.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.14, linewidth=0)

    plt.title(ENV_LABELS.get(env, env), fontweight="bold")
    plt.xlabel("# Episodes")
    plt.ylabel("Evaluation return")
    plt.grid(True, alpha=0.25)
    add_training_step_axis(axis_reference)

plt.subplot(2, 3, 6)
plt.axis("off")
legend_handles = [Line2D([0], [0], color=color, linewidth=2.0, label=label) for _, label, color in FAMILIES]
plt.legend(handles=legend_handles, loc="center", frameon=False)

plt.suptitle("Evaluation return, mean +/- std over seeds", fontweight="bold")
plt.show()
