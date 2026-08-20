import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_1 import (
    CSV_FILE,
    ENV_ORDER,
    FAMILIES,
    FIG_RES,
    RESULTS_DIR,
    best_family_curves_at_each_episode,
    load_data,
    noise_slug,
    steps_to_reach_return,
)


EXCLUDED_ENVS = {"MountainCarContinuous-v0"}

ENV_LABELS = {
    "Acrobot-v1": "Acrobot",
    "CartPole-v1": "CartPole",
    "MountainCar-v0": "Mt.Car",
    "MountainCarContinuous-v0": "Mt.Car (cont.)",
    "Pendulum-v1": "Pendulum",
}

ENV_COLORS = {
    "Acrobot-v1": "#1f77b4",
    "CartPole-v1": "#ff7f0e",
    "MountainCarContinuous-v0": "#2ca02c",
    "MountainCar-v0": "#d62728",
    "Pendulum-v1": "#9467bd",
}


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

    if not interpolated:
        return np.full_like(grid, np.nan)

    stacked = np.vstack(interpolated)
    valid_counts = np.isfinite(stacked).sum(axis=0)
    sums = np.nansum(stacked, axis=0)
    return np.divide(sums, valid_counts, out=np.full_like(grid, np.nan), where=valid_counts > 0)


def summarize_noise_level(df_noise: pd.DataFrame):
    available_envs = set(df_noise["environment_id"].unique())
    envs = [env for env in ENV_ORDER if env in available_envs and env not in EXCLUDED_ENVS]
    envs += sorted(available_envs.difference(envs).difference(EXCLUDED_ENVS))
    envs = envs[:5]

    summary_curves = {}
    for env in envs:
        best_family_curves = best_family_curves_at_each_episode(df_noise, env)
        if "kan" not in best_family_curves or "mlp" not in best_family_curves:
            continue
        if best_family_curves["kan"].empty or best_family_curves["mlp"].empty:
            continue

        merged = best_family_curves["kan"].merge(
            best_family_curves["mlp"],
            on="train_episodes",
            suffixes=("_kan", "_mlp"),
        )
        if merged.empty:
            continue

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
                sample_x.append(
                    float(
                        np.clip(
                            100.0 * (terminal_return - float(row["eval_return_mean"])) / return_span,
                            0.0,
                            100.0,
                        )
                    )
                )
            sample_y.append(100.0 * (mlp_steps - kan_steps) / mlp_steps)

        summary_curves[env] = {
            "episodes": merged["train_episodes"].to_numpy(),
            "terminal_distance": distance_from_terminal,
            "relative_improvement": merged["relative_improvement"].to_numpy(),
            "sample_terminal_distance": np.asarray(sample_x),
            "sample_reduction": np.asarray(sample_y),
        }

    return envs, summary_curves


def plot_empty(alpha: float):
    fig, ax = plt.subplots(figsize=(FIG_RES, FIG_RES * 0.35), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"No complete KAN/MLP comparison curves for reward noise alpha={alpha:g}.",
        ha="center",
        va="center",
    )
    return fig


def plot_noise_level(df_noise: pd.DataFrame, alpha: float):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    envs, summary_curves = summarize_noise_level(df_noise)
    if not summary_curves:
        return plot_empty(alpha)

    terminal_grid = np.linspace(0.0, 100.0, 201)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_RES * 2, FIG_RES * 0.7), constrained_layout=False)

    for panel_id, title in enumerate(
        ["(a) Relative improvement (%)", "(b) Improvement vs. terminal distance", "(c) Sample reduction (%)"],
    ):
        ax = axes[panel_id]
        if panel_id == 0:
            for env, curve in summary_curves.items():
                ax.plot(
                    curve["episodes"],
                    curve["relative_improvement"],
                    color=ENV_COLORS.get(env, "black"),
                    linewidth=1.3,
                    label=ENV_LABELS.get(env, env),
                )

            frames = [
                pd.DataFrame(
                    {
                        "episodes": curve["episodes"],
                        "relative_improvement": curve["relative_improvement"],
                    }
                )
                for curve in summary_curves.values()
            ]
            episode_mean = (
                pd.concat(frames, ignore_index=True)
                .groupby("episodes", as_index=False)
                .agg(relative_improvement=("relative_improvement", "mean"))
                .sort_values("episodes")
            )
            ax.plot(
                episode_mean["episodes"],
                episode_mean["relative_improvement"],
                color="black",
                linestyle="--",
                linewidth=2.2,
                label="Mean (across experiments)",
            )
            ax.set_xlabel("# Episodes (samples)")
            ax.set_ylabel("Relative improvement (%)")
            ax.axhline(0.0, color="gray", linewidth=0.8)
        elif panel_id == 1:
            for env, curve in summary_curves.items():
                order = np.argsort(curve["terminal_distance"])
                ax.plot(
                    curve["terminal_distance"][order],
                    curve["relative_improvement"][order],
                    color=ENV_COLORS.get(env, "black"),
                    linewidth=1.3,
                )
            mean_y = mean_curve_on_grid(summary_curves, "terminal_distance", "relative_improvement", terminal_grid)
            ax.plot(terminal_grid, mean_y, color="black", linestyle="--", linewidth=2.2)
            ax.set_xlabel("Distance from terminal MLP return (%)")
            ax.set_xlim(100, 0)
            ax.set_ylabel("Relative improvement (%)")
            ax.axhline(0.0, color="gray", linewidth=0.8)
        else:
            for env, curve in summary_curves.items():
                if len(curve["sample_terminal_distance"]) == 0:
                    continue
                order = np.argsort(curve["sample_terminal_distance"])
                ax.plot(
                    curve["sample_terminal_distance"][order],
                    curve["sample_reduction"][order],
                    color=ENV_COLORS.get(env, "black"),
                    linewidth=1.3,
                )
            mean_y = mean_curve_on_grid(summary_curves, "sample_terminal_distance", "sample_reduction", terminal_grid)
            ax.plot(terminal_grid, mean_y, color="black", linestyle="--", linewidth=2.2)
            ax.set_xlabel("Distance from terminal MLP return (%)")
            ax.set_xlim(100, 0)
            ax.set_ylabel("Sample reduction (%)")
            ax.axhline(0.0, color="gray", linewidth=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-5, 100)
        ax.grid(True, alpha=0.22)

    legend_envs = list(envs)
    for excluded_env in sorted(EXCLUDED_ENVS):
        if excluded_env in legend_envs:
            continue
        insert_at = len(legend_envs)
        if excluded_env == "MountainCarContinuous-v0" and "MountainCar-v0" in legend_envs:
            insert_at = legend_envs.index("MountainCar-v0") + 1
        legend_envs.insert(insert_at, excluded_env)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ENV_COLORS.get(env, "black"),
            linewidth=1.3,
            alpha=0.35 if env in EXCLUDED_ENVS else 1.0,
            label=ENV_LABELS.get(env, env),
        )
        for env in legend_envs
    ]
    legend_handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Average"))
    plt.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(legend_handles),
        handlelength=1.9,
        columnspacing=1.0,
    )
    return fig


def main() -> None:
    df = load_data()
    output_paths = []
    for alpha in sorted(df["reward_noise_alpha"].dropna().unique()):
        df_noise = df[np.isclose(df["reward_noise_alpha"], alpha)].copy()
        if df_noise.empty:
            continue
        fig = plot_noise_level(df_noise, float(alpha))
        output_path = RESULTS_DIR / f"Figure_X3_2_reward_noise_{noise_slug(alpha)}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    print(f"Read {CSV_FILE}")
    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
