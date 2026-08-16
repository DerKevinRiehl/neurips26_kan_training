from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_RESULTS_DIR = THIS_DIR.parent
CLASSIC_RESULTS_DIR = EXPERIMENT_RESULTS_DIR
FIGURE_DIR = THIS_DIR / "figures"
SUMMARY_DIR = THIS_DIR / "summaries"

MODEL_ORDER = ["kan_bspline", "kan_gaussrbf", "mlp_relu", "mlp_sigmoid"]
MODEL_LABELS = {
    "kan_bspline": "KAN B-spline",
    "kan_gaussrbf": "KAN RBF",
    "mlp_relu": "MLP ReLU",
    "mlp_sigmoid": "MLP Sigmoid",
}
MODEL_COLORS = {
    "kan_bspline": "#1f77b4",
    "kan_gaussrbf": "#2ca02c",
    "mlp_relu": "#d62728",
    "mlp_sigmoid": "#9467bd",
    "kan": "#1f77b4",
    "mlp": "#d62728",
}

USECOLS = [
    "random_seed",
    "environment_id",
    "model_setup_id",
    "model_type",
    "actor_hidden_dims",
    "critic_model_type",
    "critic_hidden_dims",
    "compare_actor_only",
    "actor_n_parameters",
    "critic_n_parameters",
    "total_n_parameters",
    "train_episodes",
    "train_env_steps_total",
    "eval_return_mean",
    "eval_return_std",
    "runtime_seconds",
]


def find_result_csv():
    csv_files = sorted(CLASSIC_RESULTS_DIR.glob("*_experiment_classic_result.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No experiment result CSV found in {CLASSIC_RESULTS_DIR}")
    return max(csv_files, key=lambda path: path.stat().st_size)


def load_results(csv_path=None, usecols=None):
    if csv_path is None:
        csv_path = find_result_csv()
    if usecols is None:
        usecols = USECOLS

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df = df[df["environment_id"].astype(str).str.contains("-v[0-9]", regex=True, na=False)].copy()

    numeric_cols = [
        "random_seed",
        "actor_n_parameters",
        "critic_n_parameters",
        "total_n_parameters",
        "train_episodes",
        "train_env_steps_total",
        "eval_return_mean",
        "eval_return_std",
        "runtime_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["random_seed"].between(0, 999) & df["train_episodes"].notna()].copy()
    df["random_seed"] = df["random_seed"].astype(int)
    df["train_episodes"] = df["train_episodes"].astype(int)
    df["model_family"] = np.where(df["model_type"].astype(str).str.startswith("kan"), "kan", "mlp")
    return df


def final_checkpoint(df):
    return int(df["train_episodes"].max())


def final_by_setup(df):
    final_ep = final_checkpoint(df)
    final = df[df["train_episodes"] == final_ep].copy()
    return (
        final.groupby(["environment_id", "model_family", "model_type", "model_setup_id"], as_index=False)
        .agg(
            eval_return_mean=("eval_return_mean", "mean"),
            eval_return_std=("eval_return_mean", "std"),
            total_n_parameters=("total_n_parameters", "first"),
            actor_n_parameters=("actor_n_parameters", "first"),
            critic_n_parameters=("critic_n_parameters", "first"),
            seeds=("random_seed", "nunique"),
        )
        .sort_values(["environment_id", "model_type", "eval_return_mean"], ascending=[True, True, False])
    )


def best_setup_per_group(df, group_cols):
    summary = final_by_setup(df)
    idx = summary.groupby(group_cols)["eval_return_mean"].idxmax()
    return summary.loc[idx].sort_values(group_cols).reset_index(drop=True)


def mean_std_curve(df, environment_id, model_setup_id):
    sub = df[(df["environment_id"] == environment_id) & (df["model_setup_id"] == model_setup_id)].copy()
    return (
        sub.groupby("train_episodes", as_index=False)
        .agg(
            eval_return_mean=("eval_return_mean", "mean"),
            eval_return_std=("eval_return_mean", "std"),
            train_env_steps_mean=("train_env_steps_total", "mean"),
            train_env_steps_std=("train_env_steps_total", "std"),
            seeds=("random_seed", "nunique"),
        )
        .sort_values("train_episodes")
    )


def safe_name(name):
    return name.replace("/", "_").replace(":", "_")
