"""Main Feynman experiment protocol for cluster runs.

This file defines the model-structure sweep and delegates execution to
experiment_runner.py.
"""


# ###########################################################################
# 1. Imports
# ###########################################################################

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ["KAN_SIMPLE_NO_AUTORUN"] = "1"

try:
    import experiment_runner as runner
    from feynman_db import list_functions
    from model_kan import KAN
    from model_mlp import MLP
except ImportError:
    from src.experiment_1_feynman import experiment_runner as runner
    from src.experiment_1_feynman.feynman_db import list_functions
    from src.experiment_1_feynman.model_kan import KAN
    from src.experiment_1_feynman.model_mlp import MLP


# ###########################################################################
# 2. Protocol Parameters
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

RUN_PROTOCOL_NOW = False
if os.environ.get("KAN_PROTOCOL_NO_AUTORUN") == "1":
    RUN_PROTOCOL_NOW = False
if os.environ.get("KAN_PROTOCOL_RUN") == "1":
    RUN_PROTOCOL_NOW = True

# Start with KAN only.
# Add "mlp_matched" after the KAN capacity sweep has run cleanly.
ACTIVE_MODEL_GROUPS = ["kan"]

KAN_HIDDEN_DIMS = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [1, 1],
    [2, 2],
    [3, 3],
]
KAN_MODEL_TYPES = ["kan_bspline", "kan_gaussrbf"]
MLP_MODEL_TYPES = ["mlp_relu", "mlp_sigmoid"]

REPRESENTATIVE_INPUT_DIM = 2
REPRESENTATIVE_OUTPUT_DIM = 1
MLP_RELATIVE_TOLERANCE = 0.25
MLP_MAX_MATCHES_PER_TARGET = 2

protocol_parameters = {
    "io": {
        "results_dir": THIS_DIR / "cluster_results",
        "results_filename": f"{RUN_TIMESTAMP}_feynman_protocol_result.csv",
        "latest_results_pointer_path": THIS_DIR / "cluster_results" / "latest_result_path.txt",
        "resume_latest_result": False,
        "resume_csv_path": None,
    },
    "experiment": {
        "data_random_seed": 42,
        "feynman_functions": [function.name for function in list_functions()],
        "random_seeds": list(range(20)),
        "train_sample_sizes": [10, 20, 30, 40, 50, 100, 200, 500, 1000],
        "test_sample_size": 1000,
    },
    "model": {
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
        "n_threads": 12,
        "torch_num_threads_per_worker": 1,
        "run_experiments_now": False,
    },
}


# ###########################################################################
# 3. Model Setup Definitions
# ###########################################################################


def hidden_label(hidden_dims: list[int] | tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in hidden_dims)


def estimate_n_parameters(model_type: str, hidden_dims: list[int]) -> int:
    model_dims = [REPRESENTATIVE_INPUT_DIM, *hidden_dims, REPRESENTATIVE_OUTPUT_DIM]
    if model_type == "mlp_relu":
        return MLP(model_dims, base_function="relu").get_n_parameters()
    if model_type == "mlp_sigmoid":
        return MLP(model_dims, base_function="sigmoid").get_n_parameters()
    if model_type == "kan_bspline":
        return KAN(
            model_dims,
            base_function="bspline",
            grid_size=protocol_parameters["model"]["kan_grid_size"],
            spline_order=protocol_parameters["model"]["kan_spline_order"],
            grid_range=protocol_parameters["model"]["kan_grid_range"],
            base_activation=protocol_parameters["model"]["kan_base_activation"],
        ).get_n_parameters()
    if model_type == "kan_gaussrbf":
        return KAN(
            model_dims,
            base_function="gaussrbf",
            grid_size=protocol_parameters["model"]["kan_grid_size"],
            spline_order=protocol_parameters["model"]["kan_spline_order"],
            grid_range=protocol_parameters["model"]["kan_grid_range"],
            base_activation=protocol_parameters["model"]["kan_base_activation"],
            use_layernorm=False,
        ).get_n_parameters()
    raise ValueError(f"Unknown model_type {model_type!r}.")


def make_setup(model_type: str, hidden_dims: list[int], group: str) -> dict[str, object]:
    n_parameters = estimate_n_parameters(model_type, hidden_dims)
    return {
        "model_setup_id": f"{model_type}_h{hidden_label(hidden_dims)}_p{n_parameters}",
        "model_type": model_type,
        "hidden_dims": hidden_dims,
        "group": group,
        "representative_input_dim": REPRESENTATIVE_INPUT_DIM,
        "representative_output_dim": REPRESENTATIVE_OUTPUT_DIM,
        "representative_n_parameters": n_parameters,
    }


KAN_MODEL_SETUPS = [
    make_setup(model_type, hidden_dims, "kan")
    for model_type in KAN_MODEL_TYPES
    for hidden_dims in KAN_HIDDEN_DIMS
]


def candidate_mlp_hidden_dims() -> list[list[int]]:
    candidates: list[list[int]] = []
    candidates.extend([[width] for width in range(1, 65)])
    candidates.extend([[width, width] for width in range(1, 65)])
    candidates.extend([[width, width, width] for width in range(1, 33)])
    candidates.extend([[max(1, width // 2), width] for width in range(2, 65)])
    candidates.extend([[width, max(1, width // 2)] for width in range(2, 65)])

    unique: dict[tuple[int, ...], list[int]] = {}
    for dims in candidates:
        unique.setdefault(tuple(dims), dims)
    return list(unique.values())


def select_parameter_matched_mlp_setups() -> list[dict[str, object]]:
    target_counts = sorted(
        {int(setup["representative_n_parameters"]) for setup in KAN_MODEL_SETUPS}
    )
    selected_dims: dict[tuple[str, tuple[int, ...]], list[int]] = {}

    for mlp_model_type in MLP_MODEL_TYPES:
        candidates = [
            (
                hidden_dims,
                estimate_n_parameters(mlp_model_type, hidden_dims),
            )
            for hidden_dims in candidate_mlp_hidden_dims()
        ]
        for target_count in target_counts:
            ranked = sorted(
                candidates,
                key=lambda item: (
                    abs(item[1] - target_count) / max(target_count, 1),
                    len(item[0]),
                    sum(item[0]),
                ),
            )
            matches = [
                item
                for item in ranked
                if abs(item[1] - target_count) / max(target_count, 1)
                <= MLP_RELATIVE_TOLERANCE
            ]
            if not matches:
                matches = ranked[:1]
            for hidden_dims, _ in matches[:MLP_MAX_MATCHES_PER_TARGET]:
                selected_dims[(mlp_model_type, tuple(hidden_dims))] = hidden_dims

    setups = [
        make_setup(mlp_model_type, hidden_dims, "mlp_matched")
        for (mlp_model_type, _), hidden_dims in selected_dims.items()
    ]
    return sorted(
        setups,
        key=lambda setup: (
            str(setup["model_type"]),
            int(setup["representative_n_parameters"]),
            str(setup["model_setup_id"]),
        ),
    )


MLP_MATCHED_MODEL_SETUPS = select_parameter_matched_mlp_setups()

ALL_MODEL_SETUPS = []
if "kan" in ACTIVE_MODEL_GROUPS:
    ALL_MODEL_SETUPS.extend(KAN_MODEL_SETUPS)
if "mlp_matched" in ACTIVE_MODEL_GROUPS:
    ALL_MODEL_SETUPS.extend(MLP_MATCHED_MODEL_SETUPS)


# ###########################################################################
# 4. Runner Configuration
# ###########################################################################


def apply_protocol_to_runner() -> None:
    runner.parameters["io"].update(protocol_parameters["io"])
    runner.parameters["experiment"].update(protocol_parameters["experiment"])
    runner.parameters["model"].update(protocol_parameters["model"])
    runner.parameters["model"]["model_setups"] = ALL_MODEL_SETUPS
    runner.parameters["training"].update(protocol_parameters["training"])
    runner.parameters["technical"].update(protocol_parameters["technical"])

    runner.RESULTS_CSV_PATH = runner.resolve_results_csv_path()

    runner.FEYNMAN_FUNCTIONS = runner.parameters["experiment"]["feynman_functions"]
    runner.DATA_RANDOM_SEED = runner.parameters["experiment"]["data_random_seed"]
    runner.RANDOM_SEEDS = runner.parameters["experiment"]["random_seeds"]
    runner.TRAIN_SAMPLE_SIZES = runner.parameters["experiment"]["train_sample_sizes"]
    runner.TEST_SAMPLE_SIZE = runner.parameters["experiment"]["test_sample_size"]

    runner.MODEL_TYPES = sorted({setup["model_type"] for setup in ALL_MODEL_SETUPS})
    runner.MODEL_HIDDEN_DIMS = []
    runner.MODEL_SETUPS = runner.get_model_setups()

    runner.TRAIN_STEPS = runner.parameters["training"]["train_steps"]
    runner.STORE_EVERY_TRAIN_STEPS = runner.parameters["training"]["store_every_train_steps"]
    runner.BATCH_SIZE = runner.parameters["training"]["batch_size"]
    runner.LEARNING_RATE = runner.parameters["training"]["learning_rate"]
    runner.WEIGHT_DECAY = runner.parameters["training"]["weight_decay"]
    runner.LOSS_FUNCTION = runner.parameters["training"]["loss_function"]
    runner.SHUFFLE_TRAINING_DATA = runner.parameters["training"]["shuffle_training_data"]

    runner.DEVICE = runner.parameters["technical"]["device"]
    runner.N_THREADS = runner.parameters["technical"]["n_threads"]
    runner.TORCH_NUM_THREADS_PER_WORKER = runner.parameters["technical"]["torch_num_threads_per_worker"]
    runner.RUN_EXPERIMENTS_NOW = False
    runner.torch.set_num_threads(runner.TORCH_NUM_THREADS_PER_WORKER)

    runner.DATASETS = runner.generate_train_test_data()


def estimate_protocol_size() -> dict[str, int]:
    n_jobs = (
        len(protocol_parameters["experiment"]["random_seeds"])
        * len(protocol_parameters["experiment"]["feynman_functions"])
        * len(protocol_parameters["experiment"]["train_sample_sizes"])
        * len(ALL_MODEL_SETUPS)
    )
    n_checkpoint_rows = n_jobs * len(runner.checkpoint_steps())
    return {
        "n_model_setups": len(ALL_MODEL_SETUPS),
        "n_jobs": n_jobs,
        "n_checkpoint_rows": n_checkpoint_rows,
    }


# ###########################################################################
# 5. Run Protocol
# ###########################################################################


def print_protocol_summary() -> None:
    size = estimate_protocol_size()
    print("Feynman protocol summary")
    print(f"Active model groups: {ACTIVE_MODEL_GROUPS}")
    print(f"Feynman functions: {len(protocol_parameters['experiment']['feynman_functions'])}")
    print(f"Random seeds: {len(protocol_parameters['experiment']['random_seeds'])}")
    print(f"Train sample sizes: {protocol_parameters['experiment']['train_sample_sizes']}")
    print(f"Model setups: {size['n_model_setups']}")
    print(f"Model trainings: {size['n_jobs']}")
    print(f"Checkpoint rows: {size['n_checkpoint_rows']}")
    print(f"Threads: {protocol_parameters['technical']['n_threads']}")
    print(f"Results file: {runner.RESULTS_CSV_PATH}")
    print("")
    print("First model setups:")
    for setup in ALL_MODEL_SETUPS[:12]:
        print(
            f"  {setup['model_setup_id']}: "
            f"{setup['model_type']} hidden={setup['hidden_dims']} "
            f"representative_params={setup['representative_n_parameters']}"
        )
    if len(ALL_MODEL_SETUPS) > 12:
        print(f"  ... {len(ALL_MODEL_SETUPS) - 12} more")


def run_protocol() -> None:
    apply_protocol_to_runner()
    print_protocol_summary()
    runner.ensure_results_csv_ready(runner.RESULTS_CSV_PATH)
    completed_result_keys = runner.load_completed_result_keys(runner.RESULTS_CSV_PATH)
    experiment_jobs = runner.build_experiment_jobs()
    runner.run_experiments_parallel(
        experiment_jobs,
        n_threads=runner.N_THREADS,
        completed_keys=completed_result_keys,
    )
    runner.finalize_results_csv(runner.RESULTS_CSV_PATH)


if RUN_PROTOCOL_NOW:
    run_protocol()
