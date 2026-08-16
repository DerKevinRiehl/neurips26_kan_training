"""CSV-producing PPO runner for Gymnasium Classic Control experiments.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Normal

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, total=None, desc=None):
        return iterable

try:
    from model_kan import KAN
    from model_mlp import MLP
except ImportError:
    from src.experiment_2_gymnasium.model_kan import KAN
    from src.experiment_2_gymnasium.model_mlp import MLP


# ###########################################################################
# 2. Parameters
# ###########################################################################

THIS_DIR = Path(__file__).resolve().parent
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
PARAMETER_SEPARATOR = "####### PARAMETERS #######"

parameters = {
    "io": {
        "results_dir": THIS_DIR / "classic_results",
        "results_filename": f"{RUN_TIMESTAMP}_experiment_classic_result.csv",
        "latest_results_pointer_path": THIS_DIR / "classic_results" / "latest_result_path.txt",
        "resume_latest_result": True,
        "resume_csv_path": None,
    },
    "experiment": {
        "environment_ids": ["CartPole-v1"],
        "render_mode": None,
        "random_seeds": [0],
        "max_train_episodes": 50,
        "eval_every_episodes": 10,
        "eval_episodes": 5,
        "eval_seed_base": 100000,
    },
    "model": {
        "model_types": ["kan_bspline", "mlp_relu"],
        "hidden_dims": [4],
        "model_setups": None,
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
        "episodes_per_update": 1,
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
        "n_threads": 1,
        "parallel_backend": "process",
        "torch_num_threads_per_worker": 1,
        "run_experiments_now": False,
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

ENVIRONMENT_IDS = parameters["experiment"]["environment_ids"]
RENDER_MODE = parameters["experiment"]["render_mode"]
RANDOM_SEEDS = parameters["experiment"]["random_seeds"]
MAX_TRAIN_EPISODES = parameters["experiment"]["max_train_episodes"]
EVAL_EVERY_EPISODES = parameters["experiment"]["eval_every_episodes"]
EVAL_EPISODES = parameters["experiment"]["eval_episodes"]
EVAL_SEED_BASE = parameters["experiment"]["eval_seed_base"]

MODEL_TYPES = parameters["model"]["model_types"]
MODEL_HIDDEN_DIMS = parameters["model"]["hidden_dims"]
COMPARE_ACTOR_ONLY = parameters["model"]["compare_actor_only"]
CRITIC_MODEL_TYPE = parameters["model"]["critic_model_type"]
CRITIC_HIDDEN_DIMS = parameters["model"]["critic_hidden_dims"]

EPISODES_PER_UPDATE = parameters["training"]["episodes_per_update"]
PPO_UPDATE_EPOCHS = parameters["training"]["ppo_update_epochs"]
LEARNING_RATE = parameters["training"]["learning_rate"]
WEIGHT_DECAY = parameters["training"]["weight_decay"]
GAMMA = parameters["training"]["gamma"]
GAE_LAMBDA = parameters["training"]["gae_lambda"]
PPO_CLIP = parameters["training"]["ppo_clip"]
VALUE_COEF = parameters["training"]["value_coef"]
ENTROPY_COEF = parameters["training"]["entropy_coef"]
MAX_GRAD_NORM = parameters["training"]["max_grad_norm"]

DEVICE = parameters["technical"]["device"]
N_THREADS = parameters["technical"]["n_threads"]
PARALLEL_BACKEND = parameters["technical"]["parallel_backend"]
TORCH_NUM_THREADS_PER_WORKER = parameters["technical"]["torch_num_threads_per_worker"]
RUN_EXPERIMENTS_NOW = parameters["technical"]["run_experiments_now"]
if os.environ.get("KAN_CLASSIC_NO_AUTORUN") == "1":
    RUN_EXPERIMENTS_NOW = False

RESULTS_LOCK = threading.Lock()


def refresh_runtime_globals() -> None:
    global RESULTS_CSV_PATH
    global ENVIRONMENT_IDS, RENDER_MODE, RANDOM_SEEDS
    global MAX_TRAIN_EPISODES, EVAL_EVERY_EPISODES, EVAL_EPISODES, EVAL_SEED_BASE
    global MODEL_TYPES, MODEL_HIDDEN_DIMS, MODEL_SETUPS
    global COMPARE_ACTOR_ONLY, CRITIC_MODEL_TYPE, CRITIC_HIDDEN_DIMS
    global EPISODES_PER_UPDATE, PPO_UPDATE_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    global GAMMA, GAE_LAMBDA, PPO_CLIP, VALUE_COEF, ENTROPY_COEF, MAX_GRAD_NORM
    global DEVICE, N_THREADS, PARALLEL_BACKEND, TORCH_NUM_THREADS_PER_WORKER
    global RUN_EXPERIMENTS_NOW

    RESULTS_CSV_PATH = resolve_results_csv_path()
    ENVIRONMENT_IDS = parameters["experiment"]["environment_ids"]
    RENDER_MODE = parameters["experiment"]["render_mode"]
    RANDOM_SEEDS = parameters["experiment"]["random_seeds"]
    MAX_TRAIN_EPISODES = parameters["experiment"]["max_train_episodes"]
    EVAL_EVERY_EPISODES = parameters["experiment"]["eval_every_episodes"]
    EVAL_EPISODES = parameters["experiment"]["eval_episodes"]
    EVAL_SEED_BASE = parameters["experiment"]["eval_seed_base"]

    MODEL_TYPES = parameters["model"]["model_types"]
    MODEL_HIDDEN_DIMS = parameters["model"]["hidden_dims"]
    MODEL_SETUPS = get_model_setups()
    COMPARE_ACTOR_ONLY = parameters["model"]["compare_actor_only"]
    CRITIC_MODEL_TYPE = parameters["model"]["critic_model_type"]
    CRITIC_HIDDEN_DIMS = parameters["model"]["critic_hidden_dims"]

    EPISODES_PER_UPDATE = parameters["training"]["episodes_per_update"]
    PPO_UPDATE_EPOCHS = parameters["training"]["ppo_update_epochs"]
    LEARNING_RATE = parameters["training"]["learning_rate"]
    WEIGHT_DECAY = parameters["training"]["weight_decay"]
    GAMMA = parameters["training"]["gamma"]
    GAE_LAMBDA = parameters["training"]["gae_lambda"]
    PPO_CLIP = parameters["training"]["ppo_clip"]
    VALUE_COEF = parameters["training"]["value_coef"]
    ENTROPY_COEF = parameters["training"]["entropy_coef"]
    MAX_GRAD_NORM = parameters["training"]["max_grad_norm"]

    DEVICE = parameters["technical"]["device"]
    N_THREADS = parameters["technical"]["n_threads"]
    PARALLEL_BACKEND = parameters["technical"]["parallel_backend"]
    TORCH_NUM_THREADS_PER_WORKER = parameters["technical"]["torch_num_threads_per_worker"]
    RUN_EXPERIMENTS_NOW = parameters["technical"]["run_experiments_now"]
    torch.set_num_threads(int(TORCH_NUM_THREADS_PER_WORKER))


def apply_runtime_parameters(runtime_parameters: dict[str, object]) -> None:
    global parameters
    parameters = runtime_parameters
    refresh_runtime_globals()


# ###########################################################################
# 3. Model and environment helpers
# ###########################################################################


@dataclass(frozen=True)
class ExperimentJob:
    random_seed: int
    environment_id: str
    model_setup_id: str
    model_type: str
    hidden_dims: tuple[int, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_setup(setup: dict[str, object]) -> dict[str, object]:
    hidden_dims = tuple(int(dim) for dim in setup["hidden_dims"])
    model_type = str(setup["model_type"])
    setup_id = str(setup.get("model_setup_id") or f"{model_type}_h{'x'.join(map(str, hidden_dims))}")
    return {
        "model_setup_id": setup_id,
        "model_type": model_type,
        "hidden_dims": hidden_dims,
    }


def get_model_setups() -> list[dict[str, object]]:
    if parameters["model"].get("model_setups") is not None:
        return [
            normalize_model_setup(setup)
            for setup in parameters["model"]["model_setups"]
        ]
    return [
        normalize_model_setup(
            {
                "model_setup_id": f"{model_type}_h{'x'.join(map(str, MODEL_HIDDEN_DIMS))}",
                "model_type": model_type,
                "hidden_dims": MODEL_HIDDEN_DIMS,
            }
        )
        for model_type in MODEL_TYPES
    ]


MODEL_SETUPS = get_model_setups()


def make_network(model_type: str, layer_dims: list[int]) -> nn.Module:
    if model_type == "mlp_relu":
        return MLP(layer_dims, base_function="relu")
    if model_type == "mlp_sigmoid":
        return MLP(layer_dims, base_function="sigmoid")
    if model_type == "mlp_gauss":
        return MLP(layer_dims, base_function="gauss")
    if model_type == "kan_bspline":
        return KAN(
            layer_dims,
            base_function="bspline",
            grid_size=parameters["model"]["kan_grid_size"],
            spline_order=parameters["model"]["kan_spline_order"],
            grid_range=parameters["model"]["kan_grid_range"],
            base_activation=parameters["model"]["kan_base_activation"],
        )
    if model_type == "kan_gaussrbf":
        return KAN(
            layer_dims,
            base_function="gaussrbf",
            grid_size=parameters["model"]["kan_grid_size"],
            spline_order=parameters["model"]["kan_spline_order"],
            grid_range=parameters["model"]["kan_grid_range"],
            base_activation=parameters["model"]["kan_base_activation"],
            use_layernorm=False,
        )
    raise ValueError(f"Unknown model_type {model_type!r}.")


def zero_output_layer(model: nn.Module) -> None:
    with torch.no_grad():
        if isinstance(model, MLP):
            for module in reversed(model.net):
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
                    return
        if isinstance(model, KAN):
            for name, parameter in model.model.layers[-1].named_parameters():
                if not parameter.requires_grad:
                    continue
                if "spline_scaler" in name:
                    continue
                parameter.zero_()
            return
    raise TypeError(f"Cannot zero output layer for {type(model)!r}.")


def observation_scale(env: gym.Env) -> torch.Tensor:
    if env.spec and env.spec.id == "CartPole-v1":
        scale = np.array([2.4, 3.0, 0.2095, 3.0], dtype=np.float32)
    else:
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        scale = np.maximum(np.abs(low), np.abs(high))
        scale[~np.isfinite(scale)] = 1.0
        scale[scale <= 0.0] = 1.0
    return torch.tensor(scale, dtype=torch.float32, device=DEVICE)


def observation_tensor(observation: np.ndarray, scale: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(observation, dtype=torch.float32, device=DEVICE)
    x = torch.clamp(x / scale, -5.0, 5.0)
    return x.unsqueeze(0)


def max_episode_steps(env: gym.Env) -> int:
    if env.spec is not None and env.spec.max_episode_steps is not None:
        return int(env.spec.max_episode_steps)
    return 1000


# ###########################################################################
# 4. PPO agent
# ###########################################################################


class PPOAgent(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        hidden_dims: tuple[int, ...],
        observation_dim: int,
        action_space: spaces.Space,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.hidden_dims = tuple(hidden_dims)

        if isinstance(action_space, spaces.Discrete):
            self.action_space_type = "discrete"
            self.action_dim = int(action_space.n)
            self.action_low = None
            self.action_high = None
            actor_output_dim = self.action_dim
        elif isinstance(action_space, spaces.Box):
            self.action_space_type = "continuous"
            self.action_dim = int(np.prod(action_space.shape))
            low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
            self.register_buffer("action_low", torch.tensor(low, dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(high, dtype=torch.float32))
            actor_output_dim = self.action_dim
            init_log_std = float(parameters["model"]["continuous_initial_log_std"])
            self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std))
        else:
            raise TypeError(f"Unsupported action space {action_space!r}.")

        self.actor = make_network(
            model_type,
            [observation_dim, *hidden_dims, actor_output_dim],
        )
        critic_type = CRITIC_MODEL_TYPE if COMPARE_ACTOR_ONLY else model_type
        critic_hidden_dims = tuple(CRITIC_HIDDEN_DIMS if COMPARE_ACTOR_ONLY else hidden_dims)
        self.critic_model_type = critic_type
        self.critic_hidden_dims = critic_hidden_dims
        self.critic = make_network(
            critic_type,
            [observation_dim, *critic_hidden_dims, 1],
        )
        zero_output_layer(self.actor)
        zero_output_layer(self.critic)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def actor_n_parameters(self) -> int:
        n_params = self.actor.get_n_parameters()
        if self.action_space_type == "continuous":
            n_params += int(self.log_std.numel())
        return n_params

    def critic_n_parameters(self) -> int:
        return self.critic.get_n_parameters()

    def total_n_parameters(self) -> int:
        return self.actor_n_parameters() + self.critic_n_parameters()

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.critic(observations).squeeze(-1)

        if self.action_space_type == "discrete":
            logits = self.actor(observations)
            distribution = Categorical(logits=logits)
            if raw_actions is None:
                if deterministic:
                    raw_actions = torch.argmax(logits, dim=-1)
                else:
                    raw_actions = distribution.sample()
            log_probs = distribution.log_prob(raw_actions)
            entropy = distribution.entropy()
            env_actions = raw_actions
            return raw_actions, env_actions, log_probs, entropy, values

        means = self.actor(observations)
        stds = torch.exp(self.log_std).expand_as(means)
        distribution = Normal(means, stds)
        if raw_actions is None:
            raw_actions = means if deterministic else distribution.sample()
        log_probs = distribution.log_prob(raw_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        squashed = torch.tanh(raw_actions)
        action_scale = (self.action_high - self.action_low) / 2.0
        action_bias = (self.action_high + self.action_low) / 2.0
        env_actions = squashed * action_scale + action_bias
        return raw_actions, env_actions, log_probs, entropy, values


# ###########################################################################
# 5. Training and evaluation
# ###########################################################################


def env_action_value(env_actions: torch.Tensor, action_space_type: str):
    action = env_actions.squeeze(0).detach().cpu().numpy()
    if action_space_type == "discrete":
        return int(action.item())
    return action


def collect_episode(
    env: gym.Env,
    agent: PPOAgent,
    scale: torch.Tensor,
    seed: int,
) -> tuple[dict[str, object], float, int]:
    observation, _ = env.reset(seed=seed)
    observations, raw_actions, log_probs, values = [], [], [], []
    rewards, dones = [], []
    episode_return = 0.0

    for _ in range(max_episode_steps(env)):
        obs_tensor = observation_tensor(observation, scale)
        with torch.no_grad():
            raw_action, env_action, log_prob, _, value = agent.get_action_and_value(obs_tensor)
        next_observation, reward, terminated, truncated, _ = env.step(
            env_action_value(env_action, agent.action_space_type)
        )
        done = terminated or truncated

        observations.append(obs_tensor.squeeze(0))
        raw_actions.append(raw_action.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(float(reward))
        dones.append(done)
        episode_return += float(reward)

        observation = next_observation
        if done:
            break

    batch = {
        "observations": torch.stack(observations),
        "raw_actions": torch.stack(raw_actions),
        "old_log_probs": torch.stack(log_probs),
        "old_values": torch.stack(values),
        "rewards": rewards,
        "dones": dones,
    }
    return batch, episode_return, len(rewards)


def concatenate_batches(batches: list[dict[str, object]]) -> dict[str, object]:
    return {
        "observations": torch.cat([batch["observations"] for batch in batches], dim=0),
        "raw_actions": torch.cat([batch["raw_actions"] for batch in batches], dim=0),
        "old_log_probs": torch.cat([batch["old_log_probs"] for batch in batches], dim=0),
        "old_values": torch.cat([batch["old_values"] for batch in batches], dim=0),
        "rewards": [reward for batch in batches for reward in batch["rewards"]],
        "dones": [done for batch in batches for done in batch["dones"]],
    }


def compute_gae(batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = batch["rewards"]
    dones = batch["dones"]
    values = batch["old_values"].detach().cpu().numpy()
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0

    for step in reversed(range(len(rewards))):
        next_nonterminal = 0.0 if dones[step] else 1.0
        delta = rewards[step] + GAMMA * next_value * next_nonterminal - values[step]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
        advantages[step] = last_gae
        next_value = values[step]

    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_tensor = advantages_tensor + batch["old_values"]
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std(unbiased=False) + 1e-8
    )
    return advantages_tensor, returns_tensor


def ppo_update(
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, object],
) -> dict[str, float]:
    observations = batch["observations"]
    raw_actions = batch["raw_actions"]
    old_log_probs = batch["old_log_probs"].detach()
    advantages, returns = compute_gae(batch)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "total_loss": 0.0,
    }

    for _ in range(PPO_UPDATE_EPOCHS):
        _, _, log_probs, entropy, values = agent.get_action_and_value(
            observations,
            raw_actions=raw_actions,
        )
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy_loss = entropy.mean()
        total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > PPO_CLIP).float().mean()
            )
        metrics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "total_loss": float(total_loss.item()),
        }

    return metrics


def evaluate_agent(
    env: gym.Env,
    agent: PPOAgent,
    scale: torch.Tensor,
    seed_start: int,
    n_episodes: int,
) -> dict[str, float]:
    returns = []
    lengths = []
    for episode_id in range(n_episodes):
        observation, _ = env.reset(seed=seed_start + episode_id)
        episode_return = 0.0
        episode_length = 0
        with torch.no_grad():
            for _ in range(max_episode_steps(env)):
                obs_tensor = observation_tensor(observation, scale)
                _, env_action, _, _, _ = agent.get_action_and_value(
                    obs_tensor,
                    deterministic=True,
                )
                observation, reward, terminated, truncated, _ = env.step(
                    env_action_value(env_action, agent.action_space_type)
                )
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break
        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        "eval_return_values": json.dumps(returns),
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns, ddof=0)),
        "eval_return_min": float(np.min(returns)),
        "eval_return_max": float(np.max(returns)),
        "eval_episode_length_values": json.dumps(lengths),
        "eval_episode_length_mean": float(np.mean(lengths)),
        "eval_env_steps_total": int(np.sum(lengths)),
    }


# ###########################################################################
# 6. CSV logging and resume
# ###########################################################################


RESULT_COLUMNS = [
    "random_seed",
    "environment_id",
    "model_setup_id",
    "model_type",
    "actor_hidden_dims",
    "critic_model_type",
    "critic_hidden_dims",
    "compare_actor_only",
    "action_space_type",
    "actor_n_parameters",
    "critic_n_parameters",
    "total_n_parameters",
    "train_episodes",
    "train_env_steps_total",
    "train_env_steps_recent",
    "train_return_recent_values",
    "train_return_recent_mean",
    "train_return_recent_std",
    "train_episode_length_recent_values",
    "train_episode_length_recent_mean",
    "eval_return_values",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_min",
    "eval_return_max",
    "eval_episode_length_values",
    "eval_episode_length_mean",
    "eval_env_steps_total",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "total_loss",
    "runtime_seconds",
]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def result_key(
    *,
    random_seed: int,
    environment_id: str,
    model_setup_id: str,
    train_episodes: int,
) -> tuple[int, str, str, int]:
    return (
        int(random_seed),
        str(environment_id),
        str(model_setup_id),
        int(train_episodes),
    )


def job_checkpoint_key(job: ExperimentJob, train_episodes: int) -> tuple[int, str, str, int]:
    return result_key(
        random_seed=job.random_seed,
        environment_id=job.environment_id,
        model_setup_id=job.model_setup_id,
        train_episodes=train_episodes,
    )


def job_identity_key(job: ExperimentJob) -> tuple[int, str, str]:
    return (
        int(job.random_seed),
        str(job.environment_id),
        str(job.model_setup_id),
    )


def row_job_identity_key(row: dict[str, object]) -> tuple[int, str, str]:
    return (
        int(row["random_seed"]),
        str(row["environment_id"]),
        str(row["model_setup_id"]),
    )


def row_key(row: dict[str, object]) -> tuple[int, str, str, int]:
    return result_key(
        random_seed=int(row["random_seed"]),
        environment_id=str(row["environment_id"]),
        model_setup_id=str(row["model_setup_id"]),
        train_episodes=int(row["train_episodes"]),
    )


def ensure_results_csv_ready(path: Path = RESULTS_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path = Path(parameters["io"]["latest_results_pointer_path"])
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(path), encoding="utf-8")

    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        data_lines = []
        for line in lines:
            if line.startswith(PARAMETER_SEPARATOR):
                break
            data_lines.append(line)
        if data_lines:
            reader = csv.reader([data_lines[0]])
            existing_header = next(reader)
            if existing_header != RESULT_COLUMNS:
                raise ValueError(f"Unexpected CSV header in {path}.")
            path.write_text("\n".join(data_lines) + "\n", encoding="utf-8")
            return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()


def append_result_rows(rows: list[dict[str, object]], path: Path | None = None) -> None:
    if not rows:
        return
    if path is None:
        path = RESULTS_CSV_PATH
    with RESULTS_LOCK:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            for row in rows:
                writer.writerow(row)


def load_completed_result_keys(path: Path = RESULTS_CSV_PATH) -> set[tuple[int, str, str, int]]:
    if not path.exists():
        return set()
    data_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(PARAMETER_SEPARATOR):
            break
        data_lines.append(line)
    completed = set()
    for row in csv.DictReader(data_lines):
        completed.add(row_key(row))
    return completed


def prune_incomplete_jobs_csv(
    jobs: list[ExperimentJob],
    path: Path = RESULTS_CSV_PATH,
) -> set[tuple[int, str, str]]:
    if not path.exists():
        return set()

    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = []
    for line in lines:
        if line.startswith(PARAMETER_SEPARATOR):
            break
        data_lines.append(line)
    if len(data_lines) <= 1:
        return set()

    rows = list(csv.DictReader(data_lines))
    active_job_keys = {job_identity_key(job) for job in jobs}
    completed_job_keys = {
        row_job_identity_key(row)
        for row in rows
        if row_job_identity_key(row) in active_job_keys
        and int(row["train_episodes"]) >= int(MAX_TRAIN_EPISODES)
    }
    kept_rows = [
        row
        for row in rows
        if row_job_identity_key(row) not in active_job_keys
        or row_job_identity_key(row) in completed_job_keys
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)

    return completed_job_keys


def finalize_results_csv(path: Path = RESULTS_CSV_PATH) -> None:
    parameter_json = json.dumps(parameters, indent=2, sort_keys=True, default=json_default)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{PARAMETER_SEPARATOR}\n")
        handle.write(parameter_json)
        handle.write("\n")


# ###########################################################################
# 7. Experiment execution
# ###########################################################################


def build_experiment_jobs() -> list[ExperimentJob]:
    jobs = []
    for environment_id in ENVIRONMENT_IDS:
        for setup in MODEL_SETUPS:
            for random_seed in RANDOM_SEEDS:
                jobs.append(
                    ExperimentJob(
                        random_seed=int(random_seed),
                        environment_id=str(environment_id),
                        model_setup_id=str(setup["model_setup_id"]),
                        model_type=str(setup["model_type"]),
                        hidden_dims=tuple(setup["hidden_dims"]),
                    )
                )
    return jobs


def train_experiment_job(
    job: ExperimentJob,
    completed_job_keys: set[tuple[int, str, str]] | None = None,
) -> list[dict[str, object]]:
    completed_job_keys = completed_job_keys or set()
    if job_identity_key(job) in completed_job_keys:
        return []

    set_seed(job.random_seed)
    train_env = gym.make(job.environment_id, render_mode=RENDER_MODE)
    eval_env = gym.make(job.environment_id, render_mode=RENDER_MODE)
    scale = observation_scale(train_env)
    observation_dim = int(np.prod(train_env.observation_space.shape))
    agent = PPOAgent(
        model_type=job.model_type,
        hidden_dims=job.hidden_dims,
        observation_dim=observation_dim,
        action_space=train_env.action_space,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    pending_batches = []
    episode_returns = []
    episode_lengths = []
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "total_loss": 0.0,
    }
    start_time = time.perf_counter()
    result_rows = []

    for train_episode in range(1, MAX_TRAIN_EPISODES + 1):
        episode_seed = job.random_seed * 1_000_000 + train_episode
        set_seed(episode_seed)
        batch, episode_return, episode_length = collect_episode(
            train_env,
            agent,
            scale,
            seed=episode_seed,
        )
        pending_batches.append(batch)
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if len(pending_batches) >= EPISODES_PER_UPDATE:
            update_batch = concatenate_batches(pending_batches)
            metrics = ppo_update(agent, optimizer, update_batch)
            pending_batches = []

        if train_episode % EVAL_EVERY_EPISODES != 0:
            continue

        eval_seed_start = EVAL_SEED_BASE + train_episode * 10_000
        eval_metrics = evaluate_agent(
            eval_env,
            agent,
            scale,
            seed_start=eval_seed_start,
            n_episodes=EVAL_EPISODES,
        )
        recent_returns = episode_returns[-EVAL_EVERY_EPISODES:]
        recent_lengths = episode_lengths[-EVAL_EVERY_EPISODES:]
        row = {
            "random_seed": job.random_seed,
            "environment_id": job.environment_id,
            "model_setup_id": job.model_setup_id,
            "model_type": job.model_type,
            "actor_hidden_dims": list(job.hidden_dims),
            "critic_model_type": agent.critic_model_type,
            "critic_hidden_dims": list(agent.critic_hidden_dims),
            "compare_actor_only": COMPARE_ACTOR_ONLY,
            "action_space_type": agent.action_space_type,
            "actor_n_parameters": agent.actor_n_parameters(),
            "critic_n_parameters": agent.critic_n_parameters(),
            "total_n_parameters": agent.total_n_parameters(),
            "train_episodes": train_episode,
            "train_env_steps_total": int(np.sum(episode_lengths)),
            "train_env_steps_recent": int(np.sum(recent_lengths)),
            "train_return_recent_values": json.dumps(recent_returns),
            "train_return_recent_mean": float(np.mean(recent_returns)),
            "train_return_recent_std": float(np.std(recent_returns, ddof=0)),
            "train_episode_length_recent_values": json.dumps(recent_lengths),
            "train_episode_length_recent_mean": float(np.mean(recent_lengths)),
            "runtime_seconds": float(time.perf_counter() - start_time),
        }
        row.update(eval_metrics)
        row.update(metrics)
        result_rows.append(row)

    train_env.close()
    eval_env.close()
    return result_rows


def train_experiment_job_worker(
    job: ExperimentJob,
    runtime_parameters: dict[str, object],
    completed_job_keys: set[tuple[int, str, str]],
) -> list[dict[str, object]]:
    apply_runtime_parameters(runtime_parameters)
    return train_experiment_job(job, completed_job_keys)


def run_experiments_parallel(
    jobs: list[ExperimentJob],
    *,
    n_threads: int,
    completed_job_keys: set[tuple[int, str, str]],
) -> None:
    pending_jobs = [
        job for job in jobs if job_identity_key(job) not in completed_job_keys
    ]
    if not pending_jobs:
        print("All classic PPO jobs are already complete.")
        return

    if n_threads <= 1:
        for job in tqdm(pending_jobs, total=len(pending_jobs), desc="classic PPO jobs"):
            rows = train_experiment_job(job, completed_job_keys)
            append_result_rows(rows)
        return

    executor_class = ProcessPoolExecutor
    if str(PARALLEL_BACKEND).lower() == "thread":
        executor_class = ThreadPoolExecutor

    with executor_class(max_workers=n_threads) as executor:
        futures = [
            executor.submit(train_experiment_job_worker, job, parameters, completed_job_keys)
            if executor_class is ProcessPoolExecutor
            else executor.submit(train_experiment_job, job, completed_job_keys)
            for job in pending_jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="classic PPO jobs"):
            rows = future.result()
            append_result_rows(rows)


def run_experiments() -> None:
    refresh_runtime_globals()
    ensure_results_csv_ready(RESULTS_CSV_PATH)
    jobs = build_experiment_jobs()
    completed_job_keys = prune_incomplete_jobs_csv(jobs, RESULTS_CSV_PATH)
    run_experiments_parallel(
        jobs,
        n_threads=N_THREADS,
        completed_job_keys=completed_job_keys,
    )
    finalize_results_csv(RESULTS_CSV_PATH)


if RUN_EXPERIMENTS_NOW:
    run_experiments()
