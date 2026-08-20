from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIG_RES = 5.5

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent

USECOLS = [
    "random_seed",
    "environment_id",
    "model_setup_id",
    "model_type",
    "reward_noise_alpha",
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
    "Acrobot-v1": "Acrobot-v1",
    "CartPole-v1": "CartPole-v1",
    "MountainCarContinuous-v0": "MountainCarContinuous-v0",
    "MountainCar-v0": "MountainCar-v0",
    "Pendulum-v1": "Pendulum-v1",
}

NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
NOISE_COLORS = [
    "#334155",
    "#2563eb",
    "#0f766e",
    "#16a34a",
    "#ca8a04",
    "#ea580c",
    "#be123c",
    "#7c3aed",
]


def find_noise_csv() -> Path:
    csv_files = sorted(RESULTS_DIR.glob("*experiment_noise_result.csv"))
    if not csv_files:
        csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}.")
    return max(csv_files, key=lambda path: path.stat().st_mtime)


CSV_FILE = find_noise_csv()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE, usecols=USECOLS, low_memory=False, on_bad_lines="skip")
    df = df[df["environment_id"].astype(str).str.contains("-v[0-9]", regex=True, na=False)].copy()
    df = df[df["model_type"].astype(str).str.startswith(("kan", "mlp"))].copy()

    for col in ["random_seed", "reward_noise_alpha", "train_episodes", "train_env_steps_total", "eval_return_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["random_seed", "reward_noise_alpha", "train_episodes", "train_env_steps_total", "eval_return_mean"]
    )
    df = df[df["random_seed"].between(0, 999)].copy()
    df["random_seed"] = df["random_seed"].astype(int)
    df["train_episodes"] = df["train_episodes"].astype(int)
    df["model_family"] = np.where(df["model_type"].astype(str).str.startswith("kan"), "kan", "mlp")
    return df


def best_mean_so_far_seed_curve(seed_level_curve: pd.DataFrame) -> pd.DataFrame:
    if seed_level_curve.empty:
        return seed_level_curve.copy()

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


def seed_curve(df: pd.DataFrame, environment_id: str, model_type: str, model_setup_id, best_so_far=True) -> pd.DataFrame:
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


def mean_std_curve(seed_level_curve: pd.DataFrame) -> pd.DataFrame:
    if seed_level_curve.empty:
        return pd.DataFrame(columns=["train_episodes", "train_env_steps_total", "eval_return_mean", "eval_return_std"])
    return (
        seed_level_curve.groupby("train_episodes", as_index=False)
        .agg(
            train_env_steps_total=("train_env_steps_total", "mean"),
            eval_return_mean=("eval_return_mean", "mean"),
            eval_return_std=("eval_return_mean", "std"),
        )
        .sort_values("train_episodes")
    )


def best_family_curve_at_each_episode(df: pd.DataFrame, environment_id: str, family: str) -> pd.DataFrame:
    candidate_curves = []
    sub = df[(df["environment_id"] == environment_id) & (df["model_family"] == family)].copy()

    for (model_type, setup_id), _ in sub.groupby(["model_type", "model_setup_id"], sort=False):
        curve = mean_std_curve(seed_curve(df, environment_id, model_type, setup_id, best_so_far=True))
        if curve.empty:
            continue
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


def best_family_curves_at_each_episode(df: pd.DataFrame, environment_id: str) -> dict[str, pd.DataFrame]:
    return {
        family: best_family_curve_at_each_episode(df, environment_id, family)
        for family, _, _ in FAMILIES
    }


def steps_to_reach_return(step_values, return_values, target_return: float) -> float:
    step_values = np.asarray(step_values, dtype=float)
    return_values = np.asarray(return_values, dtype=float)
    valid = np.isfinite(step_values) & np.isfinite(return_values)
    step_values = step_values[valid]
    return_values = return_values[valid]
    if len(step_values) == 0:
        return np.nan

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


def improvement_curve(best_family_curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "kan" not in best_family_curves or "mlp" not in best_family_curves:
        return pd.DataFrame(columns=["x", "y"])
    if best_family_curves["kan"].empty or best_family_curves["mlp"].empty:
        return pd.DataFrame(columns=["x", "y"])

    merged = best_family_curves["kan"].merge(
        best_family_curves["mlp"],
        on="train_episodes",
        suffixes=("_kan", "_mlp"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["x", "y"])

    merged["relative_improvement"] = (
        100.0
        * (merged["eval_return_mean_kan"] - merged["eval_return_mean_mlp"])
        / np.maximum(np.abs(merged["eval_return_mean_mlp"]), 1e-9)
    )
    curve = merged.sort_values("eval_return_mean_mlp")
    return pd.DataFrame(
        {
            "x": curve["eval_return_mean_mlp"].to_numpy(dtype=float),
            "y": curve["relative_improvement"].to_numpy(dtype=float),
        }
    )


def sample_reduction_curve(best_family_curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "kan" not in best_family_curves or "mlp" not in best_family_curves:
        return pd.DataFrame(columns=["x", "y"])
    if best_family_curves["kan"].empty or best_family_curves["mlp"].empty:
        return pd.DataFrame(columns=["x", "y"])

    kan_curve = best_family_curves["kan"]
    mlp_curve = best_family_curves["mlp"]
    x_values, y_values = [], []

    for _, row in mlp_curve.iterrows():
        mlp_steps = float(row["train_env_steps_total"])
        if mlp_steps <= 0:
            continue

        kan_steps = steps_to_reach_return(
            kan_curve["train_env_steps_total"].to_numpy(),
            kan_curve["eval_return_mean"].to_numpy(),
            float(row["eval_return_mean"]),
        )
        if not np.isfinite(kan_steps):
            continue

        x_values.append(float(row["eval_return_mean"]))
        y_values.append(100.0 * (mlp_steps - kan_steps) / mlp_steps)

    if not x_values:
        return pd.DataFrame(columns=["x", "y"])

    order = np.argsort(x_values)
    return pd.DataFrame(
        {
            "x": np.asarray(x_values, dtype=float)[order],
            "y": np.asarray(y_values, dtype=float)[order],
        }
    )


def ordered_noise_levels(df: pd.DataFrame) -> list[float]:
    available = sorted(float(alpha) for alpha in df["reward_noise_alpha"].dropna().unique())
    ordered = []
    for preferred_alpha in NOISE_LEVELS:
        matches = [alpha for alpha in available if np.isclose(alpha, preferred_alpha)]
        if matches:
            ordered.append(matches[0])

    ordered += [alpha for alpha in available if not any(np.isclose(alpha, kept) for kept in ordered)]
    return ordered


def noise_color_map(noise_levels: list[float]) -> dict[float, str]:
    return {
        alpha: NOISE_COLORS[index % len(NOISE_COLORS)]
        for index, alpha in enumerate(noise_levels)
    }


def plot_metric_family(ax, curves_by_alpha: dict[float, pd.DataFrame], colors_by_alpha: dict[float, str]) -> None:
    plotted = False
    for alpha, curve in curves_by_alpha.items():
        if curve.empty:
            continue

        ax.plot(
            curve["x"].to_numpy(dtype=float),
            curve["y"].to_numpy(dtype=float),
            color=colors_by_alpha.get(alpha, "black"),
            linewidth=1.25,
            alpha=0.95,
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, color="0.45")


def plot_figure(df: pd.DataFrame):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    available_envs = set(df["environment_id"].unique())
    envs = [env for env in ENV_ORDER if env in available_envs]
    envs += sorted(available_envs.difference(envs))
    envs = envs[:5]
    noise_levels = ordered_noise_levels(df)
    colors_by_alpha = noise_color_map(noise_levels)

    fig, axes = plt.subplots(
        2,
        len(envs),
        figsize=(FIG_RES * 2.0, FIG_RES * 0.7),
        constrained_layout=False,
        squeeze=False,
    )

    for col_id, env in enumerate(envs):
        best_family_curves_by_alpha = {
            alpha: best_family_curves_at_each_episode(
                df[np.isclose(df["reward_noise_alpha"], alpha)].copy(),
                env,
            )
            for alpha in noise_levels
        }

        ax = axes[0, col_id]
        plot_metric_family(
            ax,
            {
                alpha: improvement_curve(best_family_curves)
                for alpha, best_family_curves in best_family_curves_by_alpha.items()
            },
            colors_by_alpha,
        )
        ax.set_title(ENV_LABELS.get(env, env), fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(bottom=-10)
        if col_id == 0:
            ax.set_ylabel("Rel. return improvement (%)")
        if env == "MountainCarContinuous-v0":
            ax.set_yscale("symlog", linthresh=10.0)
        else:
            ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.grid(True, alpha=0.22)

        ax = axes[1, col_id]
        plot_metric_family(
            ax,
            {
                alpha: sample_reduction_curve(best_family_curves)
                for alpha, best_family_curves in best_family_curves_by_alpha.items()
            },
            colors_by_alpha,
        )
        ax.set_xlabel("Best MLP return")
        ax.set_ylim(-10, 100)
        if col_id == 0:
            ax.set_ylabel("Sample reduction (%)")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.grid(True, alpha=0.22)

    legend_handles = [
        Line2D([0], [0], color=colors_by_alpha.get(alpha, "black"), linewidth=1.5, label=f"alpha={alpha:g}")
        for alpha in noise_levels
    ]
    plt.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=len(legend_handles),
        handlelength=1.9,
        columnspacing=1.0,
    )
    return fig


df = load_data()
fig = plot_figure(df)
print(f"Read {CSV_FILE}")
plt.show()
