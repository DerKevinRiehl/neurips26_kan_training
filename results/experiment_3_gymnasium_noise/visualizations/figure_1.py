from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
    "reward_noise_std",
    "total_n_parameters",
    "train_episodes",
    "train_env_steps_total",
    "eval_return_mean",
]

PANELS = [
    ("kan_bspline", "KAN B-Spline", "#2563eb"),
    ("kan_gaussrbf", "KAN RBF", "#0f766e"),
    ("mlp_sigmoid", "MLP Sigmoid", "#9333ea"),
    ("mlp_relu", "MLP ReLU", "#dc2626"),
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

FIXED_ENV_ROWS = [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
]

FIXED_RETURN_YLIMS = {
    "Acrobot-v1": (-500, 0),
    "CartPole-v1": (0, 500),
    "MountainCarContinuous-v0": (-100, 100),
    "MountainCar-v0": (-200, 0),
    "Pendulum-v1": (-1600, 0),
}

FIXED_RELATIVE_IMPROVEMENT_YLIM = (-100, 100)
FIXED_SAMPLE_REDUCTION_YLIM = (-200, 200)

ENV_LABELS = {
    "Acrobot-v1": "Acrobot-v1",
    "CartPole-v1": "CartPole-v1",
    "MountainCarContinuous-v0": "MountainCarContinuous-v0",
    "MountainCar-v0": "MountainCar-v0",
    "Pendulum-v1": "Pendulum-v1",
}

COLUMN_TITLES = [
    "All Models",
    "All Models, Best So Far",
    "Best KAN vs. MLP",
    "Improvement vs. MLP Return",
    "Sample Reduction",
]


def find_csv_file() -> Path:
    csv_files = sorted(RESULTS_DIR.glob("*experiment_noise_result.csv"))
    if not csv_files:
        csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}.")
    return max(csv_files, key=lambda path: path.stat().st_mtime)


CSV_FILE = find_csv_file()


def noise_slug(alpha: float) -> str:
    text = f"{float(alpha):.6g}".replace("-", "m").replace(".", "p")
    return f"alpha{text}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_FILE,
        usecols=USECOLS,
        low_memory=False,
        on_bad_lines="skip",
    )
    df = df[df["environment_id"].astype(str).str.contains("-v[0-9]", regex=True, na=False)].copy()
    df = df[df["model_type"].astype(str).str.startswith(("kan", "mlp"))].copy()

    numeric_cols = [
        "random_seed",
        "reward_noise_alpha",
        "reward_noise_std",
        "total_n_parameters",
        "train_episodes",
        "train_env_steps_total",
        "eval_return_mean",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "random_seed",
            "reward_noise_alpha",
            "train_episodes",
            "train_env_steps_total",
            "eval_return_mean",
        ]
    )
    df = df[df["random_seed"].between(0, 999)].copy()
    df["random_seed"] = df["random_seed"].astype(int)
    df["train_episodes"] = df["train_episodes"].astype(int)
    df["model_family"] = np.where(df["model_type"].astype(str).str.startswith("kan"), "kan", "mlp")
    return df


def add_training_step_axis(ax, reference_curves, show_label=False):
    if not reference_curves:
        return

    reference = (
        pd.concat(reference_curves)
        .groupby("train_episodes")["train_env_steps_total"]
        .mean()
        .sort_index()
    )
    if reference.empty:
        return

    episode_values = reference.index.to_numpy(dtype=float)
    tick_ids = np.linspace(0, len(episode_values) - 1, min(4, len(episode_values))).round().astype(int)
    ticks = episode_values[np.unique(tick_ids)]

    step_labels = []
    for tick in ticks:
        steps = reference.loc[int(tick)] / 1000.0
        step_labels.append(f"{steps:.0f}" if steps >= 10.0 else f"{steps:.1f}")

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(ticks)
    top_ax.set_xticklabels(step_labels)
    top_ax.set_xlabel("# Training steps (k)" if show_label else "")
    top_ax.tick_params(axis="x", labelsize=5, pad=1)


def best_mean_so_far_seed_curve(seed_level_curve):
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


def seed_curve(df, environment_id, model_type, model_setup_id, best_so_far=False):
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


def plot_curve(ax, curve, color, label=None, linewidth=1.2):
    if curve.empty:
        return
    x = curve["train_episodes"].to_numpy()
    y = curve["eval_return_mean"].to_numpy()
    y_std = curve["eval_return_std"].fillna(0.0).to_numpy()

    ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.12, linewidth=0)


def apply_fixed_ylim(ax, environment_id: str, col_id: int) -> None:
    if col_id <= 2:
        ax.set_ylim(*FIXED_RETURN_YLIMS.get(environment_id, (-500, 500)))
    elif col_id == 3:
        ax.set_ylim(*FIXED_RELATIVE_IMPROVEMENT_YLIM)
    else:
        ax.set_ylim(*FIXED_SAMPLE_REDUCTION_YLIM)


def select_best_setup(df, group_cols, best_so_far=False):
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["model_setup_id"])

    if best_so_far:
        seed_level = (
            df.groupby(group_cols + ["model_setup_id", "random_seed", "train_episodes"], as_index=False)
            .agg(
                train_env_steps_total=("train_env_steps_total", "mean"),
                eval_return_mean=("eval_return_mean", "mean"),
            )
            .sort_values(group_cols + ["model_setup_id", "random_seed", "train_episodes"])
        )

        records = []
        for keys, setup_seed_level in seed_level.groupby(group_cols + ["model_setup_id"], sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            curve = mean_std_curve(best_mean_so_far_seed_curve(setup_seed_level))
            if curve.empty:
                continue
            record = dict(zip(group_cols + ["model_setup_id"], keys))
            record["eval_return_mean"] = curve["eval_return_mean"].mean()
            record["final_best_so_far"] = curve["eval_return_mean"].iloc[-1]
            records.append(record)

        if not records:
            return pd.DataFrame(columns=group_cols + ["model_setup_id"])

        setup_scores = pd.DataFrame(records).sort_values(
            group_cols + ["eval_return_mean", "final_best_so_far"],
            ascending=[True] * len(group_cols) + [False, False],
        )
    else:
        final_rows = (
            df.sort_values(group_cols + ["model_setup_id", "random_seed", "train_episodes"])
            .groupby(group_cols + ["model_setup_id", "random_seed"], as_index=False)
            .tail(1)
        )
        setup_scores = (
            final_rows.groupby(group_cols + ["model_setup_id", "random_seed"], as_index=False)
            .agg(eval_return_mean=("eval_return_mean", "mean"))
            .groupby(group_cols + ["model_setup_id"], as_index=False)
            .agg(eval_return_mean=("eval_return_mean", "mean"))
            .sort_values(group_cols + ["eval_return_mean"], ascending=[True] * len(group_cols) + [False])
        )

    return setup_scores.groupby(group_cols, as_index=False).head(1).copy()


def best_family_curve_at_each_episode(df, environment_id, family):
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


def best_family_curves_at_each_episode(df, environment_id):
    return {
        family: best_family_curve_at_each_episode(df, environment_id, family)
        for family, _, _ in FAMILIES
    }


def steps_to_reach_return(step_values, return_values, target_return):
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


def plot_noise_level(df_noise: pd.DataFrame, alpha: float, max_episode: float) -> Path:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    envs = FIXED_ENV_ROWS

    fig, axes = plt.subplots(
        len(envs),
        5,
        figsize=(FIG_RES * 3.0, FIG_RES * 1.9),
        constrained_layout=True,
        squeeze=False,
    )

    best_raw_model = select_best_setup(df_noise, ["environment_id", "model_type"], best_so_far=False)
    max_episode = max(float(max_episode), 1.0)

    for row_id, env in enumerate(envs):
        best_family_curves = best_family_curves_at_each_episode(df_noise, env)

        for col_id, title in enumerate(COLUMN_TITLES):
            ax = axes[row_id, col_id]
            if row_id == 0:
                ax.set_title(title, fontweight="bold")

            reference_curves = []
            has_env_data = env in set(df_noise["environment_id"].unique())

            if col_id == 0:
                for model_type, label, color in PANELS:
                    row = best_raw_model[
                        (best_raw_model["environment_id"] == env) & (best_raw_model["model_type"] == model_type)
                    ]
                    if row.empty:
                        continue
                    setup_id = row.iloc[0]["model_setup_id"]
                    curve = mean_std_curve(seed_curve(df_noise, env, model_type, setup_id, best_so_far=False))
                    reference_curves.append(curve[["train_episodes", "train_env_steps_total"]])
                    plot_curve(ax, curve, color, label=label if row_id == 0 else None)
                if row_id == 0:
                    ax.legend(
                        handles=[
                            Line2D([0], [0], color=color, linewidth=1.2, label=label)
                            for _, label, color in PANELS
                        ],
                        fontsize=5,
                        frameon=False,
                        loc="best",
                    )

            if col_id == 1:
                for model_type, _, color in PANELS:
                    row = best_raw_model[
                        (best_raw_model["environment_id"] == env) & (best_raw_model["model_type"] == model_type)
                    ]
                    if row.empty:
                        continue
                    setup_id = row.iloc[0]["model_setup_id"]
                    curve = mean_std_curve(seed_curve(df_noise, env, model_type, setup_id, best_so_far=True))
                    reference_curves.append(curve[["train_episodes", "train_env_steps_total"]])
                    plot_curve(ax, curve, color)

            if col_id == 2:
                for family, label, color in FAMILIES:
                    curve = best_family_curves.get(family, pd.DataFrame())
                    if curve.empty:
                        continue
                    reference_curves.append(curve[["train_episodes", "train_env_steps_total"]])
                    plot_curve(ax, curve, color, label=label if row_id == 0 else None, linewidth=1.4)
                if row_id == 0:
                    ax.legend(
                        handles=[
                            Line2D([0], [0], color=color, linewidth=1.4, label=label)
                            for _, label, color in FAMILIES
                        ],
                        fontsize=5,
                        frameon=False,
                        loc="best",
                    )

            if col_id == 3:
                if (
                    "kan" in best_family_curves
                    and "mlp" in best_family_curves
                    and not best_family_curves["kan"].empty
                    and not best_family_curves["mlp"].empty
                ):
                    merged = best_family_curves["kan"].merge(
                        best_family_curves["mlp"],
                        on="train_episodes",
                        suffixes=("_kan", "_mlp"),
                    )
                    if not merged.empty:
                        merged["relative_improvement"] = (
                            100.0
                            * (merged["eval_return_mean_kan"] - merged["eval_return_mean_mlp"])
                            / np.maximum(np.abs(merged["eval_return_mean_mlp"]), 1e-9)
                        )
                        improvement = merged.sort_values("eval_return_mean_mlp")
                        x = improvement["eval_return_mean_mlp"].to_numpy()
                        x_std = improvement["eval_return_std_mlp"].fillna(0.0).to_numpy()
                        y = improvement["relative_improvement"].to_numpy()
                        y_std = (
                            100.0
                            * np.sqrt(
                                improvement["eval_return_std_kan"].fillna(0.0).to_numpy() ** 2
                                + improvement["eval_return_std_mlp"].fillna(0.0).to_numpy() ** 2
                            )
                            / np.maximum(np.abs(x), 1e-9)
                        )
                        ax.plot(x, y, color="black")
                        ax.errorbar(x, y, xerr=x_std / 10, yerr=y_std / 10, color="black", alpha=0.5)
                        if env == "MountainCarContinuous-v0" and np.any(y > 0):
                            ax.set_yscale("symlog", linthresh=10.0)
                        else:
                            ax.axhline(0.0, color="gray", linewidth=0.8)

            if col_id == 4:
                if (
                    "kan" in best_family_curves
                    and "mlp" in best_family_curves
                    and not best_family_curves["kan"].empty
                    and not best_family_curves["mlp"].empty
                ):
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

                    if x_values:
                        order = np.argsort(x_values)
                        x_values = np.asarray(x_values)[order]
                        y_values = np.asarray(y_values)[order]
                        ax.plot(x_values, y_values, color="black", marker="o", markersize=1.8, linewidth=1.2)
                        ax.axhline(0.0, color="gray", linewidth=0.8)

            if not has_env_data:
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.45",
                    fontsize=6,
                )

            if col_id == 0:
                ax.set_ylabel(f"{ENV_LABELS.get(env, env)}\nReturn", fontweight="bold")
            elif col_id == 3:
                ax.set_ylabel("Rel. return improvement (%)")
            elif col_id == 4:
                ax.set_ylabel("Sample reduction (%)")
            else:
                ax.set_ylabel("")

            if col_id <= 2:
                ax.set_xlabel("# Episodes (Samples)")
                ax.tick_params(axis="x", labelbottom=True)
                ax.set_xlim(0, 6000)
                add_training_step_axis(ax, reference_curves, show_label=True)
            else:
                ax.set_xlabel("Best MLP return")
                ax.tick_params(axis="x", labelbottom=True)

            if col_id == 3:
                ax.set_ylim(bottom=-10)
            elif col_id == 4:
                ax.set_ylim(-10, 100)
            ax.grid(True, alpha=0.22)

    output_path = RESULTS_DIR / f"Figure_X3_1_reward_noise_{noise_slug(alpha)}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    df = load_data()
    max_episode = float(df["train_episodes"].max())
    output_paths = []
    for alpha in sorted(df["reward_noise_alpha"].dropna().unique()):
        df_noise = df[np.isclose(df["reward_noise_alpha"], alpha)].copy()
        if df_noise.empty:
            continue
        output_paths.append(plot_noise_level(df_noise, float(alpha), max_episode))

    print(f"Read {CSV_FILE}")
    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
