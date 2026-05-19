"""MAPPO training loop for the TorchRL FLORIS environment.

This module implements a multi-agent PPO (MAPPO) trainer using
`FlorisMultiAgentTorchRLEnv`, closely following the official TorchRL
multi-agent PPO tutorial:

    https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Key design choices:
- Each wind turbine is treated as an agent.
- A centralised critic (MAPPO) is used.
- Multiple environments run in parallel via TorchRL's ``ParallelEnv``
  (one process per env), so the OS can schedule each env on a different
  CPU core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import wandb

import torch
from torchrl.data import Composite
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_composite_lp_aggregate
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import ParallelEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from floris_torchrl_env import (
    FlorisMultiAgentTorchRLEnv,
    make_floris_torchrl_env,
)

@dataclass
class MAPPOTorchRLConfig:
    """Hyperparameters for MAPPO training."""

    # Devices
    device: torch.device = torch.device("cpu")

    # Environment / sampling
    num_envs: int = 16
    max_steps_per_episode: int = 100
    n_iters: int = 4000
    # Optimisation / PPO
    num_epochs: int = 10
    minibatch_size: int = 1024
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    gamma: float = 0.95
    lmbda: float = 0.9
    entropy_eps: float = 0.01


def _build_policy_and_critic(
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    device: torch.device,
) -> tuple[ProbabilisticActor, TensorDictModule]:
    """Construct multi-agent policy and centralised critic."""

    # Disable composite log-prob aggregation as in the tutorial
    set_composite_lp_aggregate(False).set()

    # Shared-parameter decentralised policy: each agent acts on its own observation
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralised=False,  # decentralised execution
            share_params=True,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # set bounds to match spec in`FlorisMultiAgentTorchRLEnv`.
    action_low = -torch.ones(action_dim, device=device)
    action_high = torch.ones(action_dim, device=device)

    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_low,
            "high": action_high,
        },
        spec=None,
        return_log_prob=True,
    )

    # Centralised critic (MAPPO): sees all agents' observations
    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    return policy, critic


def _collect_rollout(
    env: ParallelEnv,
    policy: ProbabilisticActor,
    max_steps: int,
    device: torch.device,
) -> TensorDict:
    """Collect one batched rollout from a ParallelEnv and assemble into a TensorDict.

    The resulting TensorDict has batch size (num_envs, max_steps) and contains:
    - ("agents", "observation")
    - ("agents", "action")
    - ("agents", "action_log_prob")
    - ("next", "agents", "observation")
    - ("next", "agents", "reward")
    - ("next", "done")
    - ("next", "terminated")
    """
    num_envs = env.batch_size[0]
    obs_spec = cast(Composite, env.observation_spec)["agents", "observation"]
    action_spec = cast(Composite, env.action_spec)["agents", "action"]
    n_agents = obs_spec.shape[-2]
    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.shape[-1]

    # Preallocate rollout tensors
    obs = torch.empty(
        num_envs, max_steps, n_agents, obs_dim, dtype=torch.float32, device=device
    )
    actions = torch.empty(
        num_envs, max_steps, n_agents, action_dim, dtype=torch.float32, device=device
    )
    action_log_prob = torch.empty(
        num_envs, max_steps, n_agents, dtype=torch.float32, device=device
    )
    rewards = torch.empty(
        num_envs, max_steps, n_agents, 1, dtype=torch.float32, device=device
    )
    done = torch.empty(
        num_envs, max_steps, 1, dtype=torch.bool, device=device
    )
    terminated = torch.empty(
        num_envs, max_steps, 1, dtype=torch.bool, device=device
    )
    next_obs = torch.empty(
        num_envs, max_steps, n_agents, obs_dim, dtype=torch.float32, device=device
    )
    baseline_power = torch.empty(num_envs, max_steps, 1, dtype=torch.float32, device=device)
    actual_power = torch.empty(num_envs, max_steps, 1, dtype=torch.float32, device=device)

    td = env.reset().to(device)

    for t in range(max_steps):
        obs[:, t] = td["agents", "observation"]

        # Sample action from policy without tracking gradients.
        # PPO will re-run the policy during optimisation; the stored
        # log-probs must be treated as fixed baselines.
        with torch.no_grad():
            td_policy = policy(td.clone())
            # Workers run on CPU; pass CPU tensordict for step, then move result back.
            # ParallelEnv expects step() to return a td with "next"; unpack it.
            step_result = env.step(td_policy.cpu()).to(device)
            td_next = step_result.get("next")

        act = td_policy["agents", "action"]
        logp = td_policy["agents", "action_log_prob"]

        actions[:, t] = act
        action_log_prob[:, t] = logp
        next_obs[:, t] = td_next["agents", "observation"]
        rewards[:, t] = td_next["agents", "reward"]
        done[:, t] = td_next["done"]
        terminated[:, t] = td_next["terminated"]
        baseline_power[:, t] = td_next["baseline_power_mw"]
        actual_power[:, t] = td_next["actual_power_mw"]

        td = td_next

    rollout_td = TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): actions,
            ("agents", "action_log_prob"): action_log_prob,
            ("next", "agents", "observation"): next_obs,
            ("next", "agents", "reward"): rewards,
            ("next", "done"): done,
            ("next", "terminated"): terminated,
            ("next", "baseline_power_mw"): baseline_power,
            ("next", "actual_power_mw"): actual_power,
        },
        batch_size=torch.Size([num_envs, max_steps]),
        device=device,
    )

    return rollout_td


def train_mappo_floris_multi_env(
    config_path: str,
    *,
    cfg: Optional[MAPPOTorchRLConfig] = None,
) -> None:
    """Train a MAPPO agent on the FLORIS TorchRL environment.

    Args:
        config_path: Path to the FLORIS YAML configuration (e.g. a farm_types/*.yaml file).
        cfg: Optional configuration object. If omitted, sensible defaults
            (including 8 parallel environments) are used.
    """
    if cfg is None:
        cfg = MAPPOTorchRLConfig()

    device = cfg.device

    # One process per env so the OS can schedule each on a different CPU core.
    def _make_single_env() -> FlorisMultiAgentTorchRLEnv:
        return make_floris_torchrl_env(
            config_path=config_path,
            max_steps=cfg.max_steps_per_episode,
            device="cpu",
        )

    env = ParallelEnv(cfg.num_envs, _make_single_env)
    try:
        # Specs are batched: (num_envs, n_agents, ...)
        obs_spec = cast(Composite, env.observation_spec)["agents", "observation"]
        action_spec = cast(Composite, env.action_spec)["agents", "action"]
        n_agents = obs_spec.shape[-2]
        obs_dim = obs_spec.shape[-1]
        action_dim = action_spec.shape[-1]

        policy, critic = _build_policy_and_critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
        )

        # PPO loss and value estimator (GAE)
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=cfg.clip_epsilon,
            entropy_coeff=cfg.entropy_eps,
            normalize_advantage=True,
        
        )
        loss_module.set_keys(
            reward=("agents", "reward"),
            action=("agents", "action"),
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        loss_module.normalize_advantage_exclude_dims = (-1,)

        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda
        )
        gae = loss_module.value_estimator

        frames_per_batch = cfg.num_envs * cfg.max_steps_per_episode

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.minibatch_size,
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

        for iter_idx in range(cfg.n_iters):
            # Data collection
            tensordict_data = _collect_rollout(
                env=env,
                policy=policy,
                max_steps=cfg.max_steps_per_episode,
                device=device,
            )

            # Expand done / terminated across the agent dimension, following the tutorial
            reward_shape = tensordict_data.get(("next", "agents", "reward")).shape
            done_expanded = (
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(reward_shape)
            )
            terminated_expanded = (
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(reward_shape)
            )

            tensordict_data.set(("next", "agents", "done"), done_expanded)
            tensordict_data.set(("next", "agents", "terminated"), terminated_expanded)

            # Compute advantages and value targets with GAE
            with torch.no_grad():
                gae(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

            # Flatten env/time dims for replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            # Track losses for logging
            cumulative_loss = 0.0
            cumulative_critic = 0.0
            cumulative_actor = 0.0

            # Optimisation epochs
            for _ in range(cfg.num_epochs):
                num_minibatches = frames_per_batch // cfg.minibatch_size
                for _ in range(num_minibatches):
                    subdata = replay_buffer.sample()
                    loss_vals = loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), max_norm=cfg.max_grad_norm
                    )
                    optim.step()
                    optim.zero_grad()

                    # for logging
                    cumulative_loss += loss_value.item()
                    cumulative_critic += loss_vals["loss_critic"].item()
                    cumulative_actor += loss_vals["loss_objective"].item()

            # calculate mean reward over  batch for logging
            rewards = tensordict_data.get(("next", "agents", "reward"))
            mean_reward = rewards.mean().item()

            mean_baseline_mw = tensordict_data.get(("next", "baseline_power_mw")).mean().item()
            mean_actual_mw = tensordict_data.get(("next", "actual_power_mw")).mean().item()

            # calculate averages over the epochs/minibatches
            total_updates = cfg.num_epochs * (frames_per_batch // cfg.minibatch_size)

            # 1. wandb logging
            wandb.log({
                "train/iteration": iter_idx + 1,
                "train/mean_reward": mean_reward,
                "train/total_loss": cumulative_loss / total_updates,
                "train/critic_loss": cumulative_critic / total_updates,
                "train/actor_loss": cumulative_actor / total_updates,
                "train/baseline_power_mw": mean_baseline_mw,
                "train/actual_power_mw": mean_actual_mw,
            })

            # 2. Keep the console print for local monitoring
            print(
                f"[MAPPO] Iteration {iter_idx + 1}/{cfg.n_iters} "
                f"- Mean Reward: {mean_reward:.4f} | Loss: {cumulative_loss / total_updates:.4f}"
            )
    finally:
        env.close()

