"""Small, Spyder-friendly draft for the Feynman sample-efficiency experiment.

This script intentionally starts small.
Run it cell by cell in Spyder, inspect the printed table and plot, and only then
increase ``TRAIN_SIZES``, ``SEEDS``, ``MAX_STEPS``, and ``FUNCTION_NAMES``.
"""

# Original sketch:
# this is an idea i had
# no main function, just runnable in spyder
# take one equation, e.g. "I.6.2"
# sample 10,000 points there, store in test_data
# sample 10,000 points there, store in train_data
# create lists of models:
# - mlp_a (relu)
# - mlp_b (sigmoid)
# - kan_a (spline)
# - kan_b (gaussrbf)
# for each model of that list, you can create multiple models of different complexity
# (number of layers and neurons), varying from too simple to complex enough to fit
# the function well.
# for each train_sample_size:
# train the model for sample from train_data and validate on test, you can check
# the training over multiple iterations and print that in console, take the peak
# the model reaches (before overfitting kicks in).

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from evaluation import evaluate_all
    from feynman_db import FeynmanFunction, get_function, list_functions
    from model_kan import KAN
    from model_mlp import MLP
except ImportError:
    from src.experiment_1_feynman.evaluation import evaluate_all
    from src.experiment_1_feynman.feynman_db import (
        FeynmanFunction,
        get_function,
        list_functions,
    )
    from src.experiment_1_feynman.model_kan import KAN
    from src.experiment_1_feynman.model_mlp import MLP


# %% Configuration

# Start with one easy-to-read equation.
# Try also: "I.6.2", "I.12.11", "II.38.3", "III.10.19", "I.9.18".
FUNCTION_NAMES = ["I.16.6"]

# Interactive defaults.
# For a first smoke test, temporarily use SEEDS = [0].
TRAIN_SIZES = [10, 20, 30, 40, 50, 100, 200, 500]
SEEDS = list(range(10))
N_TEST = 1000

# Paper-scale settings, once the draft feels trustworthy.
# TRAIN_SIZES = [10, 20, 50, 100, 200, 500, 1000]
# SEEDS = list(range(20))
# FUNCTION_NAMES = [function.name for function in list_functions()]

MODEL_NAMES = ["mlp_relu", "mlp_sigmoid", "kan_bspline", "kan_gaussrbf"]

HIDDEN_WIDTH = 16
HIDDEN_LAYERS = 2
BATCH_SIZE = 64
MAX_STEPS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
PRINT_EVERY = 100

DEVICE = "cpu"
SAVE_RESULTS_CSV = True
RESULTS_CSV_PATH = Path(__file__).resolve().parent / "results_interactive.csv"


# %% Small utility classes


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


@dataclass
class RunArtifact:
    result: dict[str, Any]
    model: nn.Module
    x_standardizer: Standardizer
    y_standardizer: Standardizer
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    training_history: list[dict[str, float]]


@dataclass
class DataSplit:
    function: FeynmanFunction
    n_train: int
    n_test: int
    seed: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_standardizer: Standardizer
    y_standardizer: Standardizer
    x_train_std: np.ndarray
    y_train_std: np.ndarray
    x_test_std: np.ndarray
    y_test_std: np.ndarray


# %% Reproducibility, data, and models


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_train_test(
    function: FeynmanFunction,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = function.sample(n_train, seed=seed)
    x_test, y_test = function.sample(n_test, seed=seed + 1_000_000)
    return x_train, y_train, x_test, y_test


def make_data_split(
    function: FeynmanFunction,
    n_train: int,
    n_test: int,
    seed: int,
    *,
    x_train_pool: np.ndarray | None = None,
    y_train_pool: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> DataSplit:
    if x_train_pool is None or y_train_pool is None or x_test is None or y_test is None:
        x_train, y_train, x_test, y_test = sample_train_test(function, n_train, n_test, seed)
    else:
        if len(x_train_pool) < n_train:
            raise ValueError(
                f"Training pool has {len(x_train_pool)} rows, but n_train={n_train}."
            )
        x_train = x_train_pool[:n_train].copy()
        y_train = y_train_pool[:n_train].copy()
        x_test = x_test.copy()
        y_test = y_test.copy()

    x_standardizer = Standardizer.fit(x_train)
    y_standardizer = Standardizer.fit(y_train)
    x_train_std = x_standardizer.transform(x_train)
    y_train_std = y_standardizer.transform(y_train)
    x_test_std = x_standardizer.transform(x_test)
    y_test_std = y_standardizer.transform(y_test)

    return DataSplit(
        function=function,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_standardizer=x_standardizer,
        y_standardizer=y_standardizer,
        x_train_std=x_train_std,
        y_train_std=y_train_std,
        x_test_std=x_test_std,
        y_test_std=y_test_std,
    )


def layer_dims(input_dim: int, output_dim: int) -> list[int]:
    hidden = [HIDDEN_WIDTH] * HIDDEN_LAYERS
    return [input_dim, *hidden, output_dim]


def make_model(model_name: str, input_dim: int, output_dim: int) -> nn.Module:
    dims = layer_dims(input_dim, output_dim)
    normalized = model_name.lower().replace("-", "_")

    if normalized in {"mlp", "mlp_silu"}:
        return MLP(dims, base_function="silu")
    if normalized in {"mlp_relu"}:
        return MLP(dims, base_function="relu")
    if normalized in {"mlp_sigmoid"}:
        return MLP(dims, base_function="sigmoid")
    if normalized in {"mlp_tanh"}:
        return MLP(dims, base_function="tanh")

    if normalized in {"kan", "kan_bspline", "efficient_kan"}:
        return KAN(
            dims,
            base_function="bspline",
            grid_size=5,
            spline_order=3,
            grid_range=(-2.0, 2.0),
            base_activation="silu",
        )
    if normalized in {"kan_fast", "kan_gaussrbf", "fast_kan"}:
        return KAN(
            dims,
            base_function="gaussrbf",
            grid_size=5,
            spline_order=3,
            grid_range=(-2.0, 2.0),
            base_activation="silu",
            use_layernorm=False,
        )
    if normalized in {"kan_bsrbf", "bsrbf_kan"}:
        return KAN(
            dims,
            base_function="bsrbf",
            grid_size=5,
            spline_order=3,
            grid_range=(-2.0, 2.0),
            base_activation="silu",
            norm_type="none",
        )

    raise ValueError(f"Unknown model_name {model_name!r}.")


def count_parameters(model: nn.Module) -> int:
    if hasattr(model, "count_parameters"):
        return int(model.count_parameters())
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# %% Training and evaluation


def evaluate_loss_std(
    model: nn.Module,
    x_std: np.ndarray,
    y_std: np.ndarray,
    loss_fn: nn.Module,
    *,
    device: str = DEVICE,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y_std, dtype=torch.float32, device=device)
        loss = loss_fn(model(x_tensor), y_tensor)
    if was_training:
        model.train()
    return float(loss.detach().cpu())


def train_model(
    model: nn.Module,
    x_train_std: np.ndarray,
    y_train_std: np.ndarray,
    x_test_std: np.ndarray,
    y_test_std: np.ndarray,
    *,
    seed: int,
    device: str = DEVICE,
) -> tuple[nn.Module, float, float, float, list[dict[str, float]]]:
    set_global_seed(seed)
    model = model.to(device)
    model.train()

    x_tensor = torch.as_tensor(x_train_std, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train_std, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=min(BATCH_SIZE, len(x_tensor)),
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    start_time = time.perf_counter()
    step = 0
    final_loss = float("nan")
    final_test_loss = float("nan")
    history: list[dict[str, float]] = []

    while step < MAX_STEPS:
        for x_batch, y_batch in loader:
            if step >= MAX_STEPS:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu())
            step += 1
            if PRINT_EVERY and step % PRINT_EVERY == 0:
                full_train_loss = evaluate_loss_std(
                    model,
                    x_train_std,
                    y_train_std,
                    loss_fn,
                    device=device,
                )
                test_loss = evaluate_loss_std(
                    model,
                    x_test_std,
                    y_test_std,
                    loss_fn,
                    device=device,
                )
                history.append(
                    {
                        "step": float(step),
                        "train_loss_std": full_train_loss,
                        "test_loss_std": test_loss,
                        "elapsed_seconds": time.perf_counter() - start_time,
                    }
                )
                print(
                    f"step {step:5d} | "
                    f"train loss {full_train_loss:.6f} | "
                    f"test loss {test_loss:.6f}"
                )

    train_seconds = time.perf_counter() - start_time
    final_loss = evaluate_loss_std(
        model,
        x_train_std,
        y_train_std,
        loss_fn,
        device=device,
    )
    final_test_loss = evaluate_loss_std(
        model,
        x_test_std,
        y_test_std,
        loss_fn,
        device=device,
    )
    if not history or int(history[-1]["step"]) != step:
        history.append(
            {
                "step": float(step),
                "train_loss_std": final_loss,
                "test_loss_std": final_test_loss,
                "elapsed_seconds": train_seconds,
            }
        )
    return model, final_loss, final_test_loss, train_seconds, history


def predict_original_scale(
    model: nn.Module,
    x_values: np.ndarray,
    x_standardizer: Standardizer,
    y_standardizer: Standardizer,
    *,
    device: str = DEVICE,
) -> np.ndarray:
    model.eval()
    x_std = x_standardizer.transform(x_values)
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=device)
        y_pred_std = model(x_tensor).detach().cpu().numpy()
    return y_standardizer.inverse_transform(y_pred_std)


def run_single_experiment(
    *,
    function_name: str,
    model_name: str,
    n_train: int,
    seed: int,
    n_test: int = N_TEST,
    device: str = DEVICE,
    data_split: DataSplit | None = None,
) -> RunArtifact:
    if data_split is None:
        function = get_function(function_name)
        data_split = make_data_split(function, n_train, n_test, seed)
    else:
        function = data_split.function

    set_global_seed(seed + 2_000_000)
    model = make_model(model_name, function.input_dim, data_split.y_train.shape[1])
    model, final_loss, final_test_loss, train_seconds, training_history = train_model(
        model,
        data_split.x_train_std,
        data_split.y_train_std,
        data_split.x_test_std,
        data_split.y_test_std,
        seed=seed + 2_000_000,
        device=device,
    )

    y_pred = predict_original_scale(
        model,
        data_split.x_test,
        data_split.x_standardizer,
        data_split.y_standardizer,
        device=device,
    )
    metrics = evaluate_all(data_split.y_test, y_pred)
    best_history = min(training_history, key=lambda row: row["test_loss_std"])

    result = {
        "function": function.name,
        "expression": function.expression,
        "input_dim": function.input_dim,
        "model": model_name,
        "n_train": data_split.n_train,
        "n_test": data_split.n_test,
        "seed": data_split.seed,
        "parameters": count_parameters(model),
        "train_seconds": train_seconds,
        "final_train_loss_std": final_loss,
        "final_test_loss_std": final_test_loss,
        "best_logged_test_loss_std": best_history["test_loss_std"],
        "best_logged_test_step": int(best_history["step"]),
        **metrics,
    }
    return RunArtifact(
        result=result,
        model=model,
        x_standardizer=data_split.x_standardizer,
        y_standardizer=data_split.y_standardizer,
        x_train=data_split.x_train,
        y_train=data_split.y_train,
        x_test=data_split.x_test,
        y_test=data_split.y_test,
        y_pred=y_pred,
        training_history=training_history,
    )


# %% Running a small learning curve


def run_learning_curve(
    function_names: list[str] = FUNCTION_NAMES,
    model_names: list[str] = MODEL_NAMES,
    train_sizes: list[int] = TRAIN_SIZES,
    seeds: list[int] = SEEDS,
    *,
    n_test: int = N_TEST,
    device: str = DEVICE,
) -> tuple[list[dict[str, Any]], list[RunArtifact]]:
    results: list[dict[str, Any]] = []
    artifacts: list[RunArtifact] = []

    total_runs = len(function_names) * len(model_names) * len(train_sizes) * len(seeds)
    run_index = 0

    for function_name in function_names:
        function = get_function(function_name)
        print("")
        print(f"Function {function.name}: {function.expression}")
        print(f"Input variables: {function.variables}")
        print(f"Domains: {function.domains}")

        for seed in seeds:
            x_train_pool, y_train_pool, x_test, y_test = sample_train_test(
                function,
                max(train_sizes),
                n_test,
                seed,
            )
            for n_train in train_sizes:
                data_split = make_data_split(
                    function,
                    n_train,
                    n_test,
                    seed,
                    x_train_pool=x_train_pool,
                    y_train_pool=y_train_pool,
                    x_test=x_test,
                    y_test=y_test,
                )
                for model_name in model_names:
                    run_index += 1
                    print("")
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"{function_name} | {model_name} | "
                        f"n_train={n_train} | seed={seed}"
                    )
                    artifact = run_single_experiment(
                        function_name=function_name,
                        model_name=model_name,
                        n_train=n_train,
                        seed=seed,
                        n_test=n_test,
                        device=device,
                        data_split=data_split,
                    )
                    artifacts.append(artifact)
                    results.append(artifact.result)
                    print_result_row(artifact.result)

    return results, artifacts


# %% Printing, aggregation, plotting, and saving


def print_result_row(result: dict[str, Any]) -> None:
    print(
        f"params={result['parameters']:6d} | "
        f"rmse={result['rmse']:.5g} | "
        f"nrmse={result['nrmse']:.5g} | "
        f"mae={result['mae']:.5g} | "
        f"rel_l2={result['relative_l2']:.5g} | "
        f"r2={result['r2']:.4f} | "
        f"test_loss={result['final_test_loss_std']:.5g} | "
        f"best_step={result['best_logged_test_step']:4d} | "
        f"time={result['train_seconds']:.2f}s"
    )


def print_results_table(results: list[dict[str, Any]]) -> None:
    header = (
        "function  model        n_train seed params  rmse       nrmse      "
        "rel_l2     r2       test_loss  best_step seconds"
    )
    print("")
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['function']:<9} "
            f"{result['model']:<12} "
            f"{result['n_train']:>7d} "
            f"{result['seed']:>4d} "
            f"{result['parameters']:>6d} "
            f"{result['rmse']:<10.4g} "
            f"{result['nrmse']:<10.4g} "
            f"{result['relative_l2']:<10.4g} "
            f"{result['r2']:<8.4f} "
            f"{result['final_test_loss_std']:<10.4g} "
            f"{result['best_logged_test_step']:<9d} "
            f"{result['train_seconds']:<7.2f}"
        )


def aggregate_metric(
    results: list[dict[str, Any]],
    *,
    metric: str = "nrmse",
    function_name: str | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    filtered = [
        result
        for result in results
        if function_name is None or result["function"] == function_name
    ]
    aggregated: dict[str, dict[int, dict[str, float]]] = {}
    for result in filtered:
        model_name = result["model"]
        n_train = int(result["n_train"])
        aggregated.setdefault(model_name, {}).setdefault(n_train, {"values": []})
        aggregated[model_name][n_train]["values"].append(float(result[metric]))

    for model_values in aggregated.values():
        for n_train, stats in model_values.items():
            values = np.array(stats["values"], dtype=np.float64)
            stats["count"] = float(len(values))
            stats["mean"] = float(values.mean())
            stats["std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            stats["std_error"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            del stats["values"]

    return aggregated


def plot_learning_curve(
    results: list[dict[str, Any]],
    *,
    metric: str = "nrmse",
    function_name: str | None = None,
    uncertainty: str = "std",
    show: bool = True,
) -> None:
    aggregated = aggregate_metric(results, metric=metric, function_name=function_name)
    plt.figure(figsize=(7.0, 4.5))
    for model_name, model_values in aggregated.items():
        train_sizes = sorted(model_values)
        means = np.array([model_values[n_train]["mean"] for n_train in train_sizes])
        errors = np.array([model_values[n_train][uncertainty] for n_train in train_sizes])
        train_sizes_array = np.array(train_sizes, dtype=np.float64)
        plt.plot(train_sizes_array, means, marker="o", label=model_name)
        plt.fill_between(
            train_sizes_array,
            means - errors,
            means + errors,
            alpha=0.15,
        )

    plt.xscale("log")
    plt.xlabel("training samples")
    plt.ylabel(metric)
    title = f"Feynman sample-efficiency curve: {metric}"
    if function_name is not None:
        title += f" on {function_name}"
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_training_history(artifact: RunArtifact, *, show: bool = True) -> None:
    history = artifact.training_history
    if not history:
        print("No training history was recorded for this artifact.")
        return

    steps = np.array([row["step"] for row in history], dtype=np.float64)
    train_losses = np.array([row["train_loss_std"] for row in history], dtype=np.float64)
    test_losses = np.array([row["test_loss_std"] for row in history], dtype=np.float64)

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(steps, train_losses, marker="o", label="train loss")
    plt.plot(steps, test_losses, marker="o", label="test loss")
    plt.xlabel("optimization steps")
    plt.ylabel("standardized MSE")
    plt.title(
        f"{artifact.result['model']} on {artifact.result['function']} "
        f"with n_train={artifact.result['n_train']} and seed={artifact.result['seed']}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def print_aggregate_table(
    results: list[dict[str, Any]],
    *,
    metric: str = "nrmse",
    function_name: str | None = None,
) -> None:
    aggregated = aggregate_metric(results, metric=metric, function_name=function_name)
    header = f"model        n_train count mean_{metric:<8} std_{metric:<8} stderr_{metric:<8}"
    print("")
    print(header)
    print("-" * len(header))
    for model_name, model_values in aggregated.items():
        for n_train in sorted(model_values):
            stats = model_values[n_train]
            print(
                f"{model_name:<12} "
                f"{n_train:>7d} "
                f"{int(stats['count']):>5d} "
                f"{stats['mean']:<13.5g} "
                f"{stats['std']:<12.5g} "
                f"{stats['std_error']:<12.5g}"
            )


def area_under_error_sample_curve(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    metric: str = "nrmse",
    function_name: str | None = None,
) -> float:
    aggregated = aggregate_metric(results, metric=metric, function_name=function_name)
    model_values = aggregated[model_name]
    train_sizes = np.array(sorted(model_values), dtype=np.float64)
    errors = np.array([model_values[int(n)]["mean"] for n in train_sizes], dtype=np.float64)
    return float(np.trapz(errors, x=np.log(train_sizes)))


def sample_threshold(
    results: list[dict[str, Any]],
    *,
    model_name: str,
    tau: float,
    metric: str = "nrmse",
    function_name: str | None = None,
) -> int | None:
    aggregated = aggregate_metric(results, metric=metric, function_name=function_name)
    model_values = aggregated[model_name]
    for n_train in sorted(model_values):
        if model_values[n_train]["mean"] <= tau:
            return n_train
    return None


def save_results_csv(results: list[dict[str, Any]], path: Path = RESULTS_CSV_PATH) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {path}")


# %% Run this cell first in Spyder


if __name__ == "__main__":
    print("Available Feynman functions:")
    print(", ".join(function.name for function in list_functions()))

    results, artifacts = run_learning_curve()
    print_results_table(results)
    print_aggregate_table(results, metric="nrmse", function_name=FUNCTION_NAMES[0])
    plot_learning_curve(results, metric="nrmse", function_name=FUNCTION_NAMES[0])

    if SAVE_RESULTS_CSV:
        save_results_csv(results)

    # The last trained model and data are easy to inspect in Spyder.
    last_artifact = artifacts[-1]
    last_model = last_artifact.model
    last_result = last_artifact.result
    # Optional diagnostic:
    # plot_training_history(last_artifact)
