"""Main noisy-reward Gymnasium Classic Control protocol for PPO cluster runs.

This file defines the environment and model sweep and delegates execution to
experiment_classic_runner.py.
"""


# ###########################################################################
# 1. Imports
# ###########################################################################

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ["KAN_CLASSIC_NO_AUTORUN"] = "1"

try:
    import experiment_classic_runner as runner
except ImportError:
    from src.experiment_3_gymnasium_noise import experiment_classic_runner as runner


# ###########################################################################
# 2. Protocol Parameters
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

RUN_PROTOCOL_NOW = __name__ == "__main__"
if os.environ.get("KAN_CLASSIC_PROTOCOL_NO_AUTORUN") == "1":
    RUN_PROTOCOL_NOW = False
if os.environ.get("KAN_CLASSIC_RUN") == "1" and __name__ == "__main__":
    RUN_PROTOCOL_NOW = True

CLASSIC_CONTROL_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]

ACTIVE_MODEL_GROUPS = ["kan", "mlp"]

KAN_MODEL_TYPES = ["kan_bspline", "kan_gaussrbf"]
KAN_HIDDEN_DIMS = [[2], [4], [8], [4, 4], [8, 8], [16, 16]]

MLP_MODEL_TYPES = ["mlp_relu", "mlp_sigmoid"]
MLP_HIDDEN_DIMS = [[16], [32], [64], [32, 32], [64, 64]]

protocol_parameters = {
    "io": {
        "results_dir": THIS_DIR / "noise_results",
        "results_filename": f"{RUN_TIMESTAMP}_experiment_noise_result.csv",
        "latest_results_pointer_path": THIS_DIR / "noise_results" / "latest_result_path.txt",
        "resume_latest_result": False,
        "resume_csv_path": None,
    },
    "experiment": {
        "environment_ids": CLASSIC_CONTROL_ENVS,
        "render_mode": None,
        "random_seeds": list(range(10)),
        "max_train_episodes": 600,
        "eval_every_episodes": 10,
        "eval_episodes": 5,
        "eval_seed_base": 100000,
    },
    "noise": {
        "reward_noise_type": "gaussian",
        "reward_noise_alphas": [0.0, 0.25, 0.5, 1.0, 2.0],
        "reward_noise_scale_mode": "unit",
        "reward_noise_scale_by_environment": {},
    },
    "model": {
        "compare_actor_only": False,
        "critic_model_type": "mlp_relu",
        "critic_hidden_dims": [32],
        "kan_grid_size": 5,
        "kan_spline_order": 3,
        "kan_grid_range": (-1.5, 1.5),
        "kan_base_activation": "silu",
        "continuous_initial_log_std": 0.0,
    },
    "training": {
        "episodes_per_update": 5,
        "ppo_update_epochs": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip": 0.20,
        "value_coef": 0.50,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.50,
    },
    "technical": {
        "device": "cpu",
        "n_threads": 15,
        "parallel_backend": "process",
        "torch_num_threads_per_worker": 1,
        "run_experiments_now": False,
    },
}


# ###########################################################################
# 3. Model Setup Definitions
# ###########################################################################


def hidden_label(hidden_dims: list[int] | tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in hidden_dims)


def make_setup(model_type: str, hidden_dims: list[int], group: str) -> dict[str, object]:
    return {
        "model_setup_id": f"{model_type}_h{hidden_label(hidden_dims)}",
        "model_type": model_type,
        "hidden_dims": hidden_dims,
        "group": group,
    }


KAN_MODEL_SETUPS = [
    [
        make_setup(model_type, hidden_dims, "kan")
        for model_type in KAN_MODEL_TYPES
    ]
    for hidden_dims in KAN_HIDDEN_DIMS
]

MLP_MODEL_SETUPS = [
    [
        make_setup(model_type, hidden_dims, "mlp")
        for model_type in MLP_MODEL_TYPES
    ]
    for hidden_dims in MLP_HIDDEN_DIMS
]

ALL_MODEL_SETUPS = []
for size_index in range(max(len(KAN_MODEL_SETUPS), len(MLP_MODEL_SETUPS))):
    if "kan" in ACTIVE_MODEL_GROUPS and size_index < len(KAN_MODEL_SETUPS):
        ALL_MODEL_SETUPS.extend(KAN_MODEL_SETUPS[size_index])
    if "mlp" in ACTIVE_MODEL_GROUPS and size_index < len(MLP_MODEL_SETUPS):
        ALL_MODEL_SETUPS.extend(MLP_MODEL_SETUPS[size_index])


# ###########################################################################
# 4. Runner Configuration
# ###########################################################################


def apply_protocol_to_runner() -> None:
    runner.parameters["io"].update(protocol_parameters["io"])
    runner.parameters["experiment"].update(protocol_parameters["experiment"])
    runner.parameters["noise"].update(protocol_parameters["noise"])
    runner.parameters["model"].update(protocol_parameters["model"])
    runner.parameters["model"]["model_setups"] = ALL_MODEL_SETUPS
    runner.parameters["training"].update(protocol_parameters["training"])
    runner.parameters["technical"].update(protocol_parameters["technical"])

    runner.RESULTS_CSV_PATH = runner.resolve_results_csv_path()

    runner.ENVIRONMENT_IDS = runner.parameters["experiment"]["environment_ids"]
    runner.RENDER_MODE = runner.parameters["experiment"]["render_mode"]
    runner.RANDOM_SEEDS = runner.parameters["experiment"]["random_seeds"]
    runner.MAX_TRAIN_EPISODES = runner.parameters["experiment"]["max_train_episodes"]
    runner.EVAL_EVERY_EPISODES = runner.parameters["experiment"]["eval_every_episodes"]
    runner.EVAL_EPISODES = runner.parameters["experiment"]["eval_episodes"]
    runner.EVAL_SEED_BASE = runner.parameters["experiment"]["eval_seed_base"]

    runner.REWARD_NOISE_TYPE = runner.parameters["noise"]["reward_noise_type"]
    runner.REWARD_NOISE_ALPHAS = runner.parameters["noise"]["reward_noise_alphas"]
    runner.REWARD_NOISE_SCALE_MODE = runner.parameters["noise"]["reward_noise_scale_mode"]
    runner.REWARD_NOISE_SCALE_BY_ENVIRONMENT = (
        runner.parameters["noise"]["reward_noise_scale_by_environment"]
    )

    runner.MODEL_TYPES = sorted({setup["model_type"] for setup in ALL_MODEL_SETUPS})
    runner.MODEL_HIDDEN_DIMS = []
    runner.MODEL_SETUPS = runner.get_model_setups()
    runner.COMPARE_ACTOR_ONLY = runner.parameters["model"]["compare_actor_only"]
    runner.CRITIC_MODEL_TYPE = runner.parameters["model"]["critic_model_type"]
    runner.CRITIC_HIDDEN_DIMS = runner.parameters["model"]["critic_hidden_dims"]

    runner.EPISODES_PER_UPDATE = runner.parameters["training"]["episodes_per_update"]
    runner.PPO_UPDATE_EPOCHS = runner.parameters["training"]["ppo_update_epochs"]
    runner.LEARNING_RATE = runner.parameters["training"]["learning_rate"]
    runner.WEIGHT_DECAY = runner.parameters["training"]["weight_decay"]
    runner.GAMMA = runner.parameters["training"]["gamma"]
    runner.GAE_LAMBDA = runner.parameters["training"]["gae_lambda"]
    runner.PPO_CLIP = runner.parameters["training"]["ppo_clip"]
    runner.VALUE_COEF = runner.parameters["training"]["value_coef"]
    runner.ENTROPY_COEF = runner.parameters["training"]["entropy_coef"]
    runner.MAX_GRAD_NORM = runner.parameters["training"]["max_grad_norm"]

    runner.DEVICE = runner.parameters["technical"]["device"]
    runner.N_THREADS = runner.parameters["technical"]["n_threads"]
    runner.PARALLEL_BACKEND = runner.parameters["technical"]["parallel_backend"]
    runner.TORCH_NUM_THREADS_PER_WORKER = runner.parameters["technical"]["torch_num_threads_per_worker"]
    runner.RUN_EXPERIMENTS_NOW = False
    runner.torch.set_num_threads(runner.TORCH_NUM_THREADS_PER_WORKER)


def estimate_protocol_size() -> dict[str, int]:
    n_jobs = (
        len(protocol_parameters["experiment"]["environment_ids"])
        * len(protocol_parameters["experiment"]["random_seeds"])
        * len(protocol_parameters["noise"]["reward_noise_alphas"])
        * len(ALL_MODEL_SETUPS)
    )
    n_checkpoints = (
        protocol_parameters["experiment"]["max_train_episodes"]
        // protocol_parameters["experiment"]["eval_every_episodes"]
    )
    return {
        "n_model_setups": len(ALL_MODEL_SETUPS),
        "n_noise_levels": len(protocol_parameters["noise"]["reward_noise_alphas"]),
        "n_jobs": n_jobs,
        "n_checkpoint_rows": n_jobs * n_checkpoints,
    }


# ###########################################################################
# 5. Run Protocol
# ###########################################################################


def print_protocol_summary() -> None:
    size = estimate_protocol_size()
    print("Gymnasium Classic Control noisy-reward PPO protocol summary")
    print(f"Active model groups: {ACTIVE_MODEL_GROUPS}")
    print(f"Environments: {protocol_parameters['experiment']['environment_ids']}")
    print(f"Render mode: {protocol_parameters['experiment']['render_mode']}")
    print(f"Random seeds: {len(protocol_parameters['experiment']['random_seeds'])}")
    print(f"Reward noise type: {protocol_parameters['noise']['reward_noise_type']}")
    print(f"Reward noise alphas: {protocol_parameters['noise']['reward_noise_alphas']}")
    print(f"Reward noise scale mode: {protocol_parameters['noise']['reward_noise_scale_mode']}")
    print(f"Noise levels: {size['n_noise_levels']}")
    print(f"Model setups: {size['n_model_setups']}")
    print(f"Model trainings: {size['n_jobs']}")
    print(f"Checkpoint rows: {size['n_checkpoint_rows']}")
    print(f"Max train episodes: {protocol_parameters['experiment']['max_train_episodes']}")
    print(f"Eval every episodes: {protocol_parameters['experiment']['eval_every_episodes']}")
    print(f"Workers: {protocol_parameters['technical']['n_threads']}")
    print(f"Parallel backend: {protocol_parameters['technical']['parallel_backend']}")
    print(f"Results file: {runner.RESULTS_CSV_PATH}")
    print("")
    print("First model setups:")
    for setup in ALL_MODEL_SETUPS[:12]:
        print(
            f"  {setup['model_setup_id']}: "
            f"{setup['model_type']} hidden={setup['hidden_dims']}"
        )
    if len(ALL_MODEL_SETUPS) > 12:
        print(f"  ... {len(ALL_MODEL_SETUPS) - 12} more")


def run_protocol() -> None:
    apply_protocol_to_runner()
    print_protocol_summary()
    runner.ensure_results_csv_ready(runner.RESULTS_CSV_PATH)
    jobs = runner.build_experiment_jobs()
    completed_job_keys = runner.prune_incomplete_jobs_csv(jobs, runner.RESULTS_CSV_PATH)
    runner.run_experiments_parallel(
        jobs,
        n_threads=runner.N_THREADS,
        completed_job_keys=completed_job_keys,
    )
    runner.finalize_results_csv(runner.RESULTS_CSV_PATH)


if RUN_PROTOCOL_NOW:
    run_protocol()
