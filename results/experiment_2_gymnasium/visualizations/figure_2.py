from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIG_RES = 5.5
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 7
plt.figure(figsize=(FIG_RES * 2, FIG_RES * 0.7), constrained_layout=True)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent
CSV_FILE = RESULTS_DIR / "20260805_2212_experiment_classic_result.csv"

USECOLS = [
    "random_seed",
    "environment_id",
    "model_setup_id",
    "model_type",
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
    # "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
]
EXCLUDED_ENVS = {"MountainCarContinuous-v0"}

ENV_LABELS = {
    "Acrobot-v1": "Acrobot-v1",
    "CartPole-v1": "CartPole-v1",
    "MountainCarContinuous-v0": "MountainCarContinuous-v0",
    "MountainCar-v0": "MountainCar-v0",
    "Pendulum-v1": "Pendulum-v1",
}

ENV_COLORS = {
    "Acrobot-v1": "#1f77b4",
    "CartPole-v1": "#ff7f0e",
    "MountainCarContinuous-v0": "#2ca02c",
    "MountainCar-v0": "#d62728",
    "Pendulum-v1": "#9467bd",
}


def best_mean_so_far_seed_curve(seed_level_curve):
    curve = seed_level_curve.sort_values(["train_episodes", "random_seed"]).copy()
    episode_mean = (
        curve.groupby("train_episodes", as_index=False)
        .agg(eval_return_mean=("eval_return_mean", "mean"))
        .sort_values("train_episodes")
    )

    best_episode_rows = []
    best_episode = None
    best_return = -np.inf
    for row in episode_mean.itertuples(index=False):
        if row.eval_return_mean > best_return:
            best_return = row.eval_return_mean
            best_episode = row.train_episodes
        best_episode_rows.append((row.train_episodes, best_episode))

    best_episode_lookup = pd.DataFrame(best_episode_rows, columns=["train_episodes", "best_episode"])
    current_steps = curve[["train_episodes", "random_seed", "train_env_steps_total"]]
    best_values = curve.rename(columns={"train_episodes": "best_episode"})[
        ["best_episode", "random_seed", "eval_return_mean"]
    ]

    return (
        current_steps.merge(best_episode_lookup, on="train_episodes", how="left")
        .merge(best_values, on=["best_episode", "random_seed"], how="left")
        .drop(columns=["best_episode"])
        .sort_values(["random_seed", "train_episodes"])
    )


def seed_curve(df, environment_id, model_type, model_setup_id, best_so_far=True):
    sub = df[
        (df["environment_id"] == environment_id)
        & (df["model_type"] == model_type)
        & (df["model_setup_id"] == model_setup_id)
    ].copy()

    curve = (
        sub.groupby(["train_episodes", "random_seed"], as_index=False)
        .agg(
            train_env_steps_total=("train_env_steps_total", "mean"),
            eval_return_mean=("eval_return_mean", "mean"),
        )
        .sort_values(["random_seed", "train_episodes"])
    )

    return best_mean_so_far_seed_curve(curve) if best_so_far else curve


def mean_std_curve(seed_level_curve):
    return (
        seed_level_curve.groupby("train_episodes", as_index=False)
        .agg(
            train_env_steps_total=("train_env_steps_total", "mean"),
            eval_return_mean=("eval_return_mean", "mean"),
            eval_return_std=("eval_return_mean", "std"),
        )
        .sort_values("train_episodes")
    )


def best_family_curve_at_each_episode(df, environment_id, family):
    candidate_curves = []
    sub = df[(df["environment_id"] == environment_id) & (df["model_family"] == family)].copy()

    for (model_type, setup_id), _ in sub.groupby(["model_type", "model_setup_id"], sort=False):
        curve = mean_std_curve(seed_curve(df, environment_id, model_type, setup_id, best_so_far=True))
        curve["model_type"] = model_type
        curve["model_setup_id"] = setup_id
        candidate_curves.append(curve)

    if not candidate_curves:
        return pd.DataFrame()

    candidates = pd.concat(candidate_curves, ignore_index=True)
    candidates = candidates.sort_values(
        ["train_episodes", "eval_return_mean", "model_type", "model_setup_id"],
        ascending=[True, False, True, True],
    )
    return candidates.groupby("train_episodes", as_index=False).head(1).sort_values("train_episodes").copy()


def best_family_curves_at_each_episode(df, environment_id):
    return {
        family: best_family_curve_at_each_episode(df, environment_id, family)
        for family, _, _ in FAMILIES
    }


def steps_to_reach_return(step_values, return_values, target_return):
    step_values = np.asarray(step_values, dtype=float)
    return_values = np.asarray(return_values, dtype=float)
    order = np.argsort(step_values)
    step_values = step_values[order]
    return_values = np.maximum.accumulate(return_values[order])

    if target_return <= return_values[0]:
        return step_values[0]
    if target_return > return_values[-1]:
        return np.nan

    idx = np.where(return_values >= target_return)[0][0]
    if idx == 0:
        return step_values[0]

    x0 = step_values[idx - 1]
    x1 = step_values[idx]
    y0 = return_values[idx - 1]
    y1 = return_values[idx]

    if abs(y1 - y0) < 1e-12:
        return x1

    weight = (target_return - y0) / (y1 - y0)
    return x0 + weight * (x1 - x0)


def mean_curve_on_grid(curves, x_key, y_key, grid):
    interpolated = []
    for curve in curves.values():
        x = np.asarray(curve[x_key], dtype=float)
        y = np.asarray(curve[y_key], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue

        ordered = (
            pd.DataFrame({"x": x[valid], "y": y[valid]})
            .groupby("x", as_index=False)
            .agg(y=("y", "mean"))
            .sort_values("x")
        )
        y_grid = np.full_like(grid, np.nan, dtype=float)
        in_range = (grid >= ordered["x"].iloc[0]) & (grid <= ordered["x"].iloc[-1])
        y_grid[in_range] = np.interp(grid[in_range], ordered["x"], ordered["y"])
        interpolated.append(y_grid)

    return np.nanmean(np.vstack(interpolated), axis=0) if interpolated else np.full_like(grid, np.nan)


df = pd.read_csv(CSV_FILE, usecols=USECOLS, low_memory=False)
df = df[df["environment_id"].astype(str).str.contains("-v[0-9]", regex=True, na=False)].copy()
df = df[df["model_type"].astype(str).str.startswith(("kan", "mlp"))].copy()

for col in ["random_seed", "train_episodes", "train_env_steps_total", "eval_return_mean"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["random_seed", "train_episodes", "train_env_steps_total", "eval_return_mean"])
df = df[df["random_seed"].between(0, 999)].copy()
df["random_seed"] = df["random_seed"].astype(int)
df["train_episodes"] = df["train_episodes"].astype(int)
df["model_family"] = np.where(df["model_type"].astype(str).str.startswith("kan"), "kan", "mlp")

available_envs = set(df["environment_id"].unique())
envs = [env for env in ENV_ORDER if env in available_envs and env not in EXCLUDED_ENVS]
envs += sorted(available_envs.difference(envs).difference(EXCLUDED_ENVS))
envs = envs[:5]

summary_curves = {}
for env in envs:
    best_family_curves = best_family_curves_at_each_episode(df, env)
    if "kan" not in best_family_curves or "mlp" not in best_family_curves:
        continue

    merged = best_family_curves["kan"].merge(
        best_family_curves["mlp"],
        on="train_episodes",
        suffixes=("_kan", "_mlp"),
    )
    merged["relative_improvement"] = (
        100.0
        * (merged["eval_return_mean_kan"] - merged["eval_return_mean_mlp"])
        / np.maximum(np.abs(merged["eval_return_mean_mlp"]), 1e-9)
    )
    mlp_returns = merged["eval_return_mean_mlp"].to_numpy(dtype=float)
    start_return = float(mlp_returns[0])
    terminal_return = float(np.nanmax(mlp_returns))
    return_span = terminal_return - start_return
    if abs(return_span) < 1e-9:
        distance_from_terminal = np.zeros_like(mlp_returns)
    else:
        distance_from_terminal = 100.0 * (terminal_return - mlp_returns) / return_span
        distance_from_terminal = np.clip(distance_from_terminal, 0.0, 100.0)

    sample_x, sample_y = [], []
    for _, row in best_family_curves["mlp"].iterrows():
        mlp_steps = float(row["train_env_steps_total"])
        if mlp_steps <= 0:
            continue

        kan_steps = steps_to_reach_return(
            best_family_curves["kan"]["train_env_steps_total"].to_numpy(),
            best_family_curves["kan"]["eval_return_mean"].to_numpy(),
            float(row["eval_return_mean"]),
        )
        if not np.isfinite(kan_steps):
            continue

        if abs(return_span) < 1e-9:
            sample_x.append(0.0)
        else:
            sample_x.append(float(np.clip(100.0 * (terminal_return - float(row["eval_return_mean"])) / return_span, 0.0, 100.0)))
        sample_y.append(100.0 * (mlp_steps - kan_steps) / mlp_steps)

    summary_curves[env] = {
        "episodes": merged["train_episodes"].to_numpy(),
        "terminal_distance": distance_from_terminal,
        "relative_improvement": merged["relative_improvement"].to_numpy(),
        "sample_terminal_distance": np.asarray(sample_x),
        "sample_reduction": np.asarray(sample_y),
    }


terminal_grid = np.linspace(0.0, 100.0, 201)

for panel_id, title in enumerate(
    ["Relative improvement (%)", "Improvement vs. terminal distance", "Sample reduction (%)"],
    start=1,
):
    plt.subplot(1, 3, panel_id)
    if panel_id == 1:
        for env, curve in summary_curves.items():
            plt.plot(
                curve["episodes"],
                curve["relative_improvement"],
                color=ENV_COLORS.get(env, "black"),
                linewidth=1.3,
                label=ENV_LABELS.get(env, env),
            )
        episode_mean = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "episodes": curve["episodes"],
                            "relative_improvement": curve["relative_improvement"],
                        }
                    )
                    for curve in summary_curves.values()
                ],
                ignore_index=True,
            )
            .groupby("episodes", as_index=False)
            .agg(relative_improvement=("relative_improvement", "mean"))
            .sort_values("episodes")
        )
        plt.plot(
            episode_mean["episodes"],
            episode_mean["relative_improvement"],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="Mean (across experiments)",
        )
        plt.xlabel("# Episodes (Samples)")
        plt.ylabel("Relative improvement (%)")
        plt.axhline(0.0, color="gray", linewidth=0.8)
    elif panel_id == 2:
        for env, curve in summary_curves.items():
            order = np.argsort(curve["terminal_distance"])
            plt.plot(
                curve["terminal_distance"][order],
                curve["relative_improvement"][order],
                color=ENV_COLORS.get(env, "black"),
                linewidth=1.3,
            )
        mean_y = mean_curve_on_grid(summary_curves, "terminal_distance", "relative_improvement", terminal_grid)
        plt.plot(terminal_grid, mean_y, color="black", linestyle="--", linewidth=2.2)
        plt.xlabel("Distance from terminal MLP return (%)")
        plt.xlim(100, 0)
        plt.ylabel("Relative improvement (%)")
        plt.axhline(0.0, color="gray", linewidth=0.8)
    else:
        for env, curve in summary_curves.items():
            if len(curve["sample_terminal_distance"]) == 0:
                continue
            order = np.argsort(curve["sample_terminal_distance"])
            plt.plot(
                curve["sample_terminal_distance"][order],
                curve["sample_reduction"][order],
                color=ENV_COLORS.get(env, "black"),
                linewidth=1.3,
            )
        mean_y = mean_curve_on_grid(summary_curves, "sample_terminal_distance", "sample_reduction", terminal_grid)
        plt.plot(terminal_grid, mean_y, color="black", linestyle="--", linewidth=2.2)
        plt.xlabel("Distance from terminal MLP return (%)")
        plt.xlim(100, 0)
        plt.ylabel("Sample reduction (%)")
        plt.axhline(0.0, color="gray", linewidth=0.8)

    plt.title(title, fontweight="bold")
    plt.ylim(-5, 100)
    plt.grid(True, alpha=0.22)

plt.subplot(1, 3, 1)
legend_handles = [
    Line2D([0], [0], color=ENV_COLORS.get(env, "black"), linewidth=1.3, label=ENV_LABELS.get(env, env))
    for env in envs
]
legend_handles += [
    Line2D(
        [0],
        [0],
        color=ENV_COLORS.get(env, "black"),
        linewidth=1.3,
        alpha=0.35,
        label=f"{ENV_LABELS.get(env, env)} (excluded)",
    )
    for env in sorted(EXCLUDED_ENVS)
]
legend_handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Mean"))
plt.legend(handles=legend_handles, fontsize=5, frameon=False, loc="best")
plt.show()
