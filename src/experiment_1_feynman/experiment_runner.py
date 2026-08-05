"""Simple CSV-producing Feynman experiment runner.

This file is intentionally section-based for Spyder.
It does not use a main function.
"""


# ###########################################################################
# 1. Imports
# ###########################################################################

from __future__ import annotations

import csv
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - useful before requirements are installed
    def tqdm(iterable, total=None, desc=None):
        return iterable

try:
    from evaluation import evaluate_all
    from feynman_db import get_function
    from model_kan import KAN
    from model_mlp import MLP
except ImportError:
    from src.experiment_1_feynman.evaluation import evaluate_all
    from src.experiment_1_feynman.feynman_db import get_function
    from src.experiment_1_feynman.model_kan import KAN
    from src.experiment_1_feynman.model_mlp import MLP


# ###########################################################################
# 2. Parameters
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

parameters = {
    "io": {
        "results_dir": THIS_DIR,
        "results_filename": f"{RUN_TIMESTAMP}_result.csv",
        "latest_results_pointer_path": THIS_DIR / "latest_result_path.txt",
        "resume_latest_result": True,
        "resume_csv_path": None,
    },
    "experiment": {
        "data_random_seed": 42,
        "feynman_functions": ["I.16.6"],
        "random_seeds": list(range(10)),
        "train_sample_sizes": [10, 20, 30, 40, 50, 100, 200, 500, 1000],
        "test_sample_size": 1000,
    },
    "model": {
        # "model_types": ["mlp_relu", "mlp_sigmoid", "kan_bspline", "kan_gaussrbf"],
        "model_types": ["kan_bspline"],
        "hidden_dims": [2],
        "kan_grid_size": 5,
        "kan_spline_order": 3,
        "kan_grid_range": (-2.0, 2.0),
        "kan_base_activation": "silu",
    },
    "training": {
        "train_steps": 1500,
        "store_every_train_steps": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "loss_function": "mse",
        "shuffle_training_data": False,
    },
    "technical": {
        "device": "cpu",
        "n_threads": 5,
        "torch_num_threads_per_worker": 1,
        "run_experiments_now": True,
    },
}


def resolve_results_csv_path() -> Path:
    resume_csv_path = parameters["io"].get("resume_csv_path")
    if resume_csv_path is not None:
        path = Path(resume_csv_path)
    else:
        pointer_path = Path(parameters["io"]["latest_results_pointer_path"])
        resume_latest = bool(parameters["io"]["resume_latest_result"])
        if resume_latest and pointer_path.exists():
            candidate = Path(pointer_path.read_text(encoding="utf-8").strip())
            if candidate.exists():
                path = candidate
            else:
                path = Path(parameters["io"]["results_dir"]) / parameters["io"]["results_filename"]
        else:
            path = Path(parameters["io"]["results_dir"]) / parameters["io"]["results_filename"]

    parameters["io"]["resolved_results_csv_path"] = str(path)
    return path


RESULTS_CSV_PATH = resolve_results_csv_path()

FEYNMAN_FUNCTIONS = parameters["experiment"]["feynman_functions"]
DATA_RANDOM_SEED = parameters["experiment"]["data_random_seed"]
RANDOM_SEEDS = parameters["experiment"]["random_seeds"]
TRAIN_SAMPLE_SIZES = parameters["experiment"]["train_sample_sizes"]
TEST_SAMPLE_SIZE = parameters["experiment"]["test_sample_size"]

MODEL_TYPES = parameters["model"]["model_types"]
MODEL_HIDDEN_DIMS = parameters["model"]["hidden_dims"]

TRAIN_STEPS = parameters["training"]["train_steps"]
STORE_EVERY_TRAIN_STEPS = parameters["training"]["store_every_train_steps"]
BATCH_SIZE = parameters["training"]["batch_size"]
LEARNING_RATE = parameters["training"]["learning_rate"]
WEIGHT_DECAY = parameters["training"]["weight_decay"]
LOSS_FUNCTION = parameters["training"]["loss_function"]
SHUFFLE_TRAINING_DATA = parameters["training"]["shuffle_training_data"]

DEVICE = parameters["technical"]["device"]
N_THREADS = parameters["technical"]["n_threads"]
TORCH_NUM_THREADS_PER_WORKER = parameters["technical"]["torch_num_threads_per_worker"]
RUN_EXPERIMENTS_NOW = parameters["technical"]["run_experiments_now"]

if os.environ.get("KAN_SIMPLE_NO_AUTORUN") == "1":
    RUN_EXPERIMENTS_NOW = False
torch.set_num_threads(TORCH_NUM_THREADS_PER_WORKER)


# ###########################################################################
# 3. Generate Train and Test Data
# ###########################################################################


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


@dataclass(frozen=True)
class ExperimentJob:
    random_seed: int
    feynman_function: str
    model_setup_id: str
    model_type: str
    hidden_dims: tuple[int, ...]
    train_sample_size: int


MODEL_INIT_LOCK = threading.Lock()
CSV_WRITE_LOCK = threading.Lock()


def generate_train_test_data() -> dict[str, dict[str, np.ndarray]]:
    datasets: dict[str, dict[str, np.ndarray]] = {}
    max_train_size = max(TRAIN_SAMPLE_SIZES)
    np.random.seed(DATA_RANDOM_SEED)

    for function_index, feynman_function in enumerate(FEYNMAN_FUNCTIONS):
        function = get_function(feynman_function)
        train_seed = DATA_RANDOM_SEED + function_index * 10_000
        test_seed = DATA_RANDOM_SEED + 1_000_000 + function_index * 10_000
        x_train_pool, y_train_pool = function.sample(max_train_size, seed=train_seed)
        x_test, y_test = function.sample(TEST_SAMPLE_SIZE, seed=test_seed)
        datasets[feynman_function] = {
            "x_train_pool": x_train_pool,
            "y_train_pool": y_train_pool,
            "x_test": x_test,
            "y_test": y_test,
        }

    return datasets


DATASETS = generate_train_test_data()


# ###########################################################################
# 4. Code to Run Single Experiment and Parallel Experiments
# ###########################################################################


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_setup(setup: dict[str, object]) -> dict[str, object]:
    hidden_dims = tuple(int(dim) for dim in setup["hidden_dims"])
    model_type = str(setup["model_type"])
    setup_id = str(setup.get("model_setup_id") or f"{model_type}_{'x'.join(map(str, hidden_dims))}")
    return {
        "model_setup_id": setup_id,
        "model_type": model_type,
        "hidden_dims": hidden_dims,
    }


def get_model_setups() -> list[dict[str, object]]:
    if "model_setups" in parameters["model"]:
        return [
            normalize_model_setup(setup)
            for setup in parameters["model"]["model_setups"]
        ]
    return [
        normalize_model_setup(
            {
                "model_setup_id": f"{model_type}_{'x'.join(map(str, MODEL_HIDDEN_DIMS))}",
                "model_type": model_type,
                "hidden_dims": MODEL_HIDDEN_DIMS,
            }
        )
        for model_type in MODEL_TYPES
    ]


MODEL_SETUPS = get_model_setups()


def model_dims_for(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
) -> list[int]:
    hidden = MODEL_HIDDEN_DIMS if hidden_dims is None else list(hidden_dims)
    return [input_dim, *hidden, output_dim]


def stable_model_seed(job: ExperimentJob) -> int:
    return int(job.random_seed)


def make_model(model_type: str, model_dims: list[int]) -> nn.Module:
    if model_type == "mlp_relu":
        return MLP(model_dims, base_function="relu")
    if model_type == "mlp_sigmoid":
        return MLP(model_dims, base_function="sigmoid")
    if model_type == "mlp_gauss":
        return MLP(model_dims, base_function="gauss")
    if model_type == "mlp_silu":
        return MLP(model_dims, base_function="silu")
    if model_type == "kan_bspline":
        return KAN(
            model_dims,
            base_function="bspline",
            grid_size=parameters["model"]["kan_grid_size"],
            spline_order=parameters["model"]["kan_spline_order"],
            grid_range=parameters["model"]["kan_grid_range"],
            base_activation=parameters["model"]["kan_base_activation"],
        )
    if model_type == "kan_gaussrbf":
        return KAN(
            model_dims,
            base_function="gaussrbf",
            grid_size=parameters["model"]["kan_grid_size"],
            spline_order=parameters["model"]["kan_spline_order"],
            grid_range=parameters["model"]["kan_grid_range"],
            base_activation=parameters["model"]["kan_base_activation"],
            use_layernorm=False,
        )
    raise ValueError(f"Unknown model_type {model_type!r}.")


def make_loss_function(loss_function: str) -> nn.Module:
    normalized = loss_function.strip().lower().replace("-", "_")
    if normalized in {"mse", "mse_loss", "mean_squared_error"}:
        return nn.MSELoss()
    if normalized in {"mae", "l1", "l1_loss", "mean_absolute_error"}:
        return nn.L1Loss()
    if normalized in {"huber", "smooth_l1", "smooth_l1_loss"}:
        return nn.SmoothL1Loss()
    raise ValueError(
        "Unknown loss_function "
        f"{loss_function!r}. Supported: mse, mae, huber."
    )


def evaluate_loss_std(
    model: nn.Module,
    x_std: np.ndarray,
    y_std: np.ndarray,
    loss_fn: nn.Module,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.as_tensor(y_std, dtype=torch.float32, device=DEVICE)
        loss = loss_fn(model(x_tensor), y_tensor)
    if was_training:
        model.train()
    return float(loss.detach().cpu())


def predict_original_scale(
    model: nn.Module,
    x_values: np.ndarray,
    x_standardizer: Standardizer,
    y_standardizer: Standardizer,
) -> np.ndarray:
    model.eval()
    x_std = x_standardizer.transform(x_values)
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=DEVICE)
        y_pred_std = model(x_tensor).detach().cpu().numpy()
    return y_standardizer.inverse_transform(y_pred_std)


def make_result_row(
    job: ExperimentJob,
    model: nn.Module,
    model_dims: list[int],
    train_steps: int,
    train_loss: float,
    test_loss: float,
    runtime_seconds: float,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    metrics = evaluate_all(y_test, y_pred)
    row: dict[str, object] = {
        "random_seed": job.random_seed,
        "feynman_function": job.feynman_function,
        "model_setup_id": job.model_setup_id,
        "model_type": job.model_type,
        "model_dims": json.dumps(model_dims),
        "model_n_parameters": model.get_n_parameters(),
        "train_sample_size": job.train_sample_size,
        "train_steps": train_steps,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "runtime_seconds": runtime_seconds,
    }
    for metric_name, metric_value in metrics.items():
        row[f"test_eval_{metric_name}"] = metric_value
    return row


def checkpoint_steps() -> list[int]:
    steps = list(range(STORE_EVERY_TRAIN_STEPS, TRAIN_STEPS + 1, STORE_EVERY_TRAIN_STEPS))
    if not steps or steps[-1] != TRAIN_STEPS:
        steps.append(TRAIN_STEPS)
    return steps


def result_key(
    *,
    random_seed: int,
    feynman_function: str,
    model_setup_id: str,
    model_type: str,
    train_sample_size: int,
    train_steps: int,
) -> tuple[int, str, str, str, int, int]:
    return (
        int(random_seed),
        str(feynman_function),
        str(model_setup_id),
        str(model_type),
        int(train_sample_size),
        int(train_steps),
    )


def job_checkpoint_key(job: ExperimentJob, train_steps: int) -> tuple[int, str, str, str, int, int]:
    return result_key(
        random_seed=job.random_seed,
        feynman_function=job.feynman_function,
        model_setup_id=job.model_setup_id,
        model_type=job.model_type,
        train_sample_size=job.train_sample_size,
        train_steps=train_steps,
    )


def run_single_experiment(
    job: ExperimentJob,
    completed_keys: set[tuple[int, str, str, str, int, int]] | None = None,
) -> list[dict[str, object]]:
    completed_keys = completed_keys or set()
    missing_steps = [
        step for step in checkpoint_steps() if job_checkpoint_key(job, step) not in completed_keys
    ]
    if not missing_steps:
        return []
    missing_steps_set = set(missing_steps)

    function = get_function(job.feynman_function)
    dataset = DATASETS[job.feynman_function]

    x_train = dataset["x_train_pool"][: job.train_sample_size]
    y_train = dataset["y_train_pool"][: job.train_sample_size]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    x_standardizer = Standardizer.fit(x_train)
    y_standardizer = Standardizer.fit(y_train)
    x_train_std = x_standardizer.transform(x_train)
    y_train_std = y_standardizer.transform(y_train)
    x_test_std = x_standardizer.transform(x_test)
    y_test_std = y_standardizer.transform(y_test)

    model_dims = model_dims_for(function.input_dim, y_train.shape[1], job.hidden_dims)
    model_seed = stable_model_seed(job)
    with MODEL_INIT_LOCK:
        set_global_seed(model_seed)
        model = make_model(job.model_type, model_dims).to(DEVICE)

    train_dataset = TensorDataset(
        torch.as_tensor(x_train_std, dtype=torch.float32),
        torch.as_tensor(y_train_std, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        shuffle=SHUFFLE_TRAINING_DATA,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = make_loss_function(LOSS_FUNCTION)
    rows: list[dict[str, object]] = []
    step = 0
    start_time = time.perf_counter()

    while step < TRAIN_STEPS:
        for x_batch, y_batch in train_loader:
            if step >= TRAIN_STEPS:
                break

            model.train()
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss_fn(model(x_batch), y_batch)
            batch_loss.backward()
            optimizer.step()

            step += 1
            should_store = step in missing_steps_set
            if should_store:
                train_loss = evaluate_loss_std(model, x_train_std, y_train_std, loss_fn)
                test_loss = evaluate_loss_std(model, x_test_std, y_test_std, loss_fn)
                runtime_seconds = time.perf_counter() - start_time
                y_pred = predict_original_scale(
                    model,
                    x_test,
                    x_standardizer,
                    y_standardizer,
                )
                row = make_result_row(
                    job,
                    model,
                    model_dims,
                    step,
                    train_loss,
                    test_loss,
                    runtime_seconds,
                    y_test,
                    y_pred,
                )
                rows.append(row)
                append_result_rows([row])

    return rows


def build_experiment_jobs() -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []
    for random_seed in RANDOM_SEEDS:
        for feynman_function in FEYNMAN_FUNCTIONS:
            for train_sample_size in TRAIN_SAMPLE_SIZES:
                for model_setup in MODEL_SETUPS:
                    jobs.append(
                        ExperimentJob(
                            random_seed=random_seed,
                            feynman_function=feynman_function,
                            model_setup_id=str(model_setup["model_setup_id"]),
                            model_type=str(model_setup["model_type"]),
                            hidden_dims=tuple(model_setup["hidden_dims"]),
                            train_sample_size=train_sample_size,
                        )
                    )
    return jobs


def run_experiments_parallel(
    jobs: list[ExperimentJob],
    n_threads: int = N_THREADS,
    completed_keys: set[tuple[int, str, str, str, int, int]] | None = None,
) -> list[dict[str, object]]:
    completed_keys = completed_keys or load_completed_result_keys(RESULTS_CSV_PATH)
    all_rows: list[dict[str, object]] = []

    if n_threads <= 1:
        for job in tqdm(jobs, desc="experiments"):
            all_rows.extend(run_single_experiment(job, completed_keys))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(run_single_experiment, job, completed_keys) for job in jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="experiments"):
                all_rows.extend(future.result())

    return sorted(
        all_rows,
        key=lambda row: (
            row["random_seed"],
            row["feynman_function"],
            row["train_sample_size"],
            row["model_setup_id"],
            row["model_type"],
            row["train_steps"],
        ),
    )


# ###########################################################################
# 5. Store All Results to results.csv
# ###########################################################################


RESULT_COLUMNS = [
    "random_seed",
    "feynman_function",
    "model_setup_id",
    "model_type",
    "model_dims",
    "model_n_parameters",
    "train_sample_size",
    "train_steps",
    "train_loss",
    "test_loss",
    "runtime_seconds",
    "test_eval_mse",
    "test_eval_rmse",
    "test_eval_mae",
    "test_eval_nrmse",
    "test_eval_relative_l2",
    "test_eval_r2",
    "test_eval_max_absolute_error",
]

PARAMETER_SEPARATOR = "####### PARAMETERS #######"


def current_results_csv_path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else Path(RESULTS_CSV_PATH)


def write_latest_results_pointer(path: Path) -> None:
    pointer_path = Path(parameters["io"]["latest_results_pointer_path"])
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(path), encoding="utf-8")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parameters_json() -> str:
    return json.dumps(parameters, indent=2, sort_keys=True, default=json_default)


def data_lines_from_csv(path: Path | None = None) -> list[str]:
    path = current_results_csv_path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines: list[str] = []
    for line in lines:
        if line.startswith(PARAMETER_SEPARATOR):
            break
        data_lines.append(line)
    return data_lines


def read_existing_result_rows(path: Path | None = None) -> list[dict[str, str]]:
    path = current_results_csv_path(path)
    data_lines = data_lines_from_csv(path)
    if not data_lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    return [row for row in reader if row.get("random_seed")]


def strip_parameter_block(path: Path | None = None) -> None:
    path = current_results_csv_path(path)
    if not path.exists():
        return
    data_lines = data_lines_from_csv(path)
    path.write_text("\n".join(data_lines).rstrip() + "\n", encoding="utf-8")


def ensure_results_csv_ready(path: Path | None = None) -> None:
    path = current_results_csv_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_latest_results_pointer(path)
    if path.exists():
        strip_parameter_block(path)
        data_lines = data_lines_from_csv(path)
        if data_lines:
            existing_header = next(csv.reader([data_lines[0]]))
            if existing_header != RESULT_COLUMNS:
                rows = read_existing_result_rows(path)
                write_results_csv(rows, path, include_parameters=False)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()


def result_key_from_row(row: dict[str, object]) -> tuple[int, str, str, str, int, int]:
    model_setup_id = row.get("model_setup_id") or f"{row['model_type']}_{row.get('model_dims', '')}"
    return result_key(
        random_seed=int(row["random_seed"]),
        feynman_function=str(row["feynman_function"]),
        model_setup_id=str(model_setup_id),
        model_type=str(row["model_type"]),
        train_sample_size=int(row["train_sample_size"]),
        train_steps=int(row["train_steps"]),
    )


def load_completed_result_keys(
    path: Path | None = None,
) -> set[tuple[int, str, str, str, int, int]]:
    path = current_results_csv_path(path)
    return {result_key_from_row(row) for row in read_existing_result_rows(path)}


def append_result_rows(rows: list[dict[str, object]], path: Path | None = None) -> None:
    if not rows:
        return
    path = current_results_csv_path(path)
    with CSV_WRITE_LOCK:
        ensure_results_csv_ready(path)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writerows(rows)


def sort_result_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["random_seed"]),
            str(row["feynman_function"]),
            int(row["train_sample_size"]),
            str(row.get("model_setup_id", "")),
            str(row["model_type"]),
            int(row["train_steps"]),
        ),
    )


def write_results_csv(
    rows: list[dict[str, object]],
    path: Path | None = None,
    *,
    include_parameters: bool = True,
) -> None:
    path = current_results_csv_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(sort_result_rows(rows))
        if include_parameters:
            handle.write(f"{PARAMETER_SEPARATOR}\n")
            handle.write(parameters_json())
            handle.write("\n")


def finalize_results_csv(path: Path | None = None) -> None:
    path = current_results_csv_path(path)
    rows = read_existing_result_rows(path)
    write_results_csv(rows, path, include_parameters=True)
    print(f"Saved {len(rows)} checkpoint rows to {path}.")


def save_results_csv(rows: list[dict[str, object]], path: Path | None = None) -> None:
    path = current_results_csv_path(path)
    write_results_csv(rows, path, include_parameters=True)
    print(f"Saved {len(rows)} checkpoint rows to {path}.")


if RUN_EXPERIMENTS_NOW:
    ensure_results_csv_ready(RESULTS_CSV_PATH)
    completed_result_keys = load_completed_result_keys(RESULTS_CSV_PATH)
    experiment_jobs = build_experiment_jobs()
    experiment_results = run_experiments_parallel(
        experiment_jobs,
        n_threads=N_THREADS,
        completed_keys=completed_result_keys,
    )
    finalize_results_csv(RESULTS_CSV_PATH)
