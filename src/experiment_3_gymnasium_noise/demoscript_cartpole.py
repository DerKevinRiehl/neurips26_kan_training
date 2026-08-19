"""Small paired PPO CartPole demo for KAN and MLP policies.

Run this file in Spyder.
It alternates KAN and MLP training episodes and renders evaluation episodes.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_kan import KAN
from model_mlp import MLP


# ###########################################################################
# Parameters
# ###########################################################################

ENV_ID = "CartPole-v1"
SEED = 42
TOTAL_EPISODES_PER_MODEL = 600  # Later use 600.
PRINT_EVERY_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
EVAL_EPISODES = 5
RENDER_EVALUATION = True
RENDER_DELAY_SECONDS = 0.01

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.20
PPO_UPDATE_EPOCHS = 4
LEARNING_RATE = 3e-4
VALUE_COEF = 0.50
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.50

COMPARE_ACTOR_ONLY = False
KAN_SIZE_CANDIDATES = [[2], [4], [8], [4, 4], [8, 8], [16, 16]]
MLP_SIZE_CANDIDATES = [[16], [32], [64], [32, 32], [64, 64]]
KAN_HIDDEN_DIMS = [4]
KAN_BASE_FUNCTION = "bspline"
MLP_HIDDEN_DIMS = [32]
MLP_BASE_FUNCTION = "relu"
CRITIC_HIDDEN_DIMS = [32]

CLASSIC_CONTROL_ENVS = [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
]
BOX2D_ENVS = [
    "BipedalWalker-v3",
    "CarRacing-v3",
    "LunarLander-v3",
]


# ###########################################################################
# Helper code
# ###########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_scale = torch.tensor([2.4, 3.0, 0.2095, 3.0], dtype=torch.float32, device=device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_tensor(observation):
    x = torch.tensor(observation, dtype=torch.float32, device=device)
    x = torch.clamp(x / obs_scale, -1.0, 1.0)
    return x.unsqueeze(0)


def zero_output_layer(model):
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


def make_actor(model_kind, observation_dim, action_dim):
    if model_kind == "kan":
        actor = KAN(
            [observation_dim, *KAN_HIDDEN_DIMS, action_dim],
            base_function=KAN_BASE_FUNCTION,
            grid_size=5,
            spline_order=3,
            grid_range=(-1.5, 1.5),
            base_activation="silu",
        ).to(device)
    elif model_kind == "mlp":
        actor = MLP(
            [observation_dim, *MLP_HIDDEN_DIMS, action_dim],
            base_function=MLP_BASE_FUNCTION,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_kind {model_kind!r}.")
    zero_output_layer(actor)
    return actor


def make_critic(model_kind, observation_dim):
    if COMPARE_ACTOR_ONLY or model_kind == "mlp":
        critic = MLP(
            [observation_dim, *CRITIC_HIDDEN_DIMS, 1],
            base_function=MLP_BASE_FUNCTION,
        ).to(device)
    else:
        critic = KAN(
            [observation_dim, *KAN_HIDDEN_DIMS, 1],
            base_function=KAN_BASE_FUNCTION,
            grid_size=5,
            spline_order=3,
            grid_range=(-1.5, 1.5),
            base_activation="silu",
        ).to(device)
    zero_output_layer(critic)
    return critic


def actor_critic_values(actor, critic, observations, actions=None):
    logits = actor(observations)
    distribution = Categorical(logits=logits)
    if actions is None:
        actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy()
    values = critic(observations).squeeze(-1)
    return actions, log_probs, entropy, values


def collect_episode(env, actor, critic, seed):
    observation, _ = env.reset(seed=seed)
    observations, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    episode_return = 0.0

    for _ in range(MAX_STEPS_PER_EPISODE):
        obs_tensor = observation_tensor(observation)
        with torch.no_grad():
            action, log_prob, _, value = actor_critic_values(actor, critic, obs_tensor)
        next_observation, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        observations.append(obs_tensor.squeeze(0))
        actions.append(action.squeeze(0))
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
        "actions": torch.stack(actions),
        "old_log_probs": torch.stack(log_probs),
        "old_values": torch.stack(values),
        "rewards": rewards,
        "dones": dones,
    }
    return batch, episode_return


def compute_gae(batch):
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

    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = advantages + batch["old_values"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages, returns


def ppo_update(actor, critic, optimizer, batch):
    observations = batch["observations"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"].detach()
    advantages, returns = compute_gae(batch)

    final_loss = 0.0
    for _ in range(PPO_UPDATE_EPOCHS):
        _, log_probs, entropy, values = actor_critic_values(actor, critic, observations, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
        optimizer.step()
        final_loss = float(loss.item())

    return final_loss


def evaluate_agent(env, actor, critic, seed_start, render_first_episode=False):
    returns = []
    for episode_id in range(EVAL_EPISODES):
        observation, _ = env.reset(seed=seed_start + episode_id)
        episode_return = 0.0
        with torch.no_grad():
            for _ in range(MAX_STEPS_PER_EPISODE):
                obs_tensor = observation_tensor(observation)
                logits = actor(obs_tensor).squeeze(0)
                action = int(torch.argmax(logits).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if render_first_episode and episode_id == 0:
                    env.render()
                    time.sleep(RENDER_DELAY_SECONDS)
                if terminated or truncated:
                    break
        returns.append(episode_return)
    return float(np.mean(returns)), float(np.std(returns, ddof=0))


def make_agent(label, model_kind, observation_dim, action_dim):
    set_seed(SEED)
    actor = make_actor(model_kind, observation_dim, action_dim)
    set_seed(SEED + 1)
    critic = make_critic(model_kind, observation_dim)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LEARNING_RATE,
    )
    return {
        "label": label,
        "kind": model_kind,
        "actor": actor,
        "critic": critic,
        "optimizer": optimizer,
        "returns": [],
        "losses": [],
    }


# ###########################################################################
# Run demo
# ###########################################################################

set_seed(SEED)
train_envs = {"KAN": gym.make(ENV_ID), "MLP": gym.make(ENV_ID)}
eval_env = gym.make(ENV_ID, render_mode="human" if RENDER_EVALUATION else None)
observation_dim = train_envs["KAN"].observation_space.shape[0]
action_dim = train_envs["KAN"].action_space.n

agents = [
    make_agent("KAN", "kan", observation_dim, action_dim),
    make_agent("MLP", "mlp", observation_dim, action_dim),
]

print("Starting paired PPO CartPole demo.")
print(f"Device: {device}")
print("CartPole reward: +1 per survived step, max episode return 500.")
print(f"Compare actor only: {COMPARE_ACTOR_ONLY}")
for agent in agents:
    actor_params = agent["actor"].get_n_parameters()
    critic_params = agent["critic"].get_n_parameters()
    print(f"{agent['label']} actor params: {actor_params} | critic params: {critic_params}")

for episode in range(1, TOTAL_EPISODES_PER_MODEL + 1):
    for agent in agents:
        env = train_envs[agent["label"]]
        env.action_space.seed(SEED + episode)
        set_seed(SEED + 1000000 + episode)
        batch, episode_return = collect_episode(
            env,
            agent["actor"],
            agent["critic"],
            seed=SEED + episode,
        )
        loss = ppo_update(agent["actor"], agent["critic"], agent["optimizer"], batch)
        agent["returns"].append(episode_return)
        agent["losses"].append(loss)

    if episode % PRINT_EVERY_EPISODES == 0 or episode == 1:
        print("")
        print(f"After {episode} episodes per model")
        for agent in agents:
            eval_mean, eval_std = evaluate_agent(
                eval_env,
                agent["actor"],
                agent["critic"],
                seed_start=SEED + 100000 + episode * 100,
                render_first_episode=RENDER_EVALUATION,
            )
            recent_train = np.mean(agent["returns"][-PRINT_EVERY_EPISODES:])
            recent_loss = np.mean(agent["losses"][-PRINT_EVERY_EPISODES:])
            print(
                f"{agent['label']:>3} | "
                f"recent train return {recent_train:7.2f} | "
                f"eval return {eval_mean:7.2f} +/- {eval_std:6.2f} | "
                f"loss {recent_loss: .4f}"
            )

for env in train_envs.values():
    env.close()
eval_env.close()
print("")
print("Done.")
