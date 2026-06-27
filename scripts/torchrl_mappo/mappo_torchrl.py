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
import time
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

import os


def save_checkpoint(
    path: str,
    iteration: int,
    loss_module: "ClipPPOLoss",
    optim: torch.optim.Optimizer,
) -> None:
    """Save model + optimizer state so training can be resumed later."""
    checkpoint = {
        "iteration": iteration,
        "loss_module_state_dict": loss_module.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    loss_module: "ClipPPOLoss",
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load model + optimizer state. Returns the iteration to resume from."""
    checkpoint = torch.load(path, map_location=device)
    loss_module.load_state_dict(checkpoint["loss_module_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

@dataclass
class MAPPOTorchRLConfig:
    """Hyperparameters for MAPPO training."""

    # Devices
    device: torch.device = torch.device("cpu")

    # Environment / sampling
    num_envs: int = 32
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
) -> tuple[TensorDict, dict[str, float]]:
    """Collect one batched rollout from a ParallelEnv and assemble into a TensorDict.

    Returns:
        A tuple of (rollout_td, timing) where timing contains cumulative
        wall-clock time (seconds) spent in the policy forward pass vs. in
        env.step() (the latter is dominated by FLORIS computation and blocks
        until every parallel worker finishes its step).
    """
    num_envs = env.batch_size[0]
    obs_spec = cast(Composite, env.observation_spec)["agents", "observation"]
    action_spec = cast(Composite, env.action_spec)["agents", "action"]
    n_agents = obs_spec.shape[-2]
    obs_dim = obs_spec.shape[-1]
    action_dim = action_spec.shape[-1]

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
    done = torch.empty(num_envs, max_steps, 1, dtype=torch.bool, device=device)
    terminated = torch.empty(num_envs, max_steps, 1, dtype=torch.bool, device=device)
    next_obs = torch.empty(
        num_envs, max_steps, n_agents, obs_dim, dtype=torch.float32, device=device
    )
    baseline_power = torch.empty(num_envs, max_steps, 1, dtype=torch.float32, device=device)
    actual_power = torch.empty(num_envs, max_steps, 1, dtype=torch.float32, device=device)

    policy_time = 0.0
    env_step_time = 0.0

    td = env.reset().to(device)

    for t in range(max_steps):
        obs[:, t] = td["agents", "observation"]

        with torch.no_grad():
            t0 = time.perf_counter()
            td_policy = policy(td.clone())
            t1 = time.perf_counter()
            step_result = env.step(td_policy.cpu()).to(device)
            t2 = time.perf_counter()
            td_next = step_result.get("next")

        policy_time += t1 - t0
        env_step_time += t2 - t1

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

    timing = {
        "policy_time_s": policy_time,
        "env_step_time_s": env_step_time,
        "policy_time_per_step_s": policy_time / max_steps,
        "env_step_time_per_step_s": env_step_time / max_steps,
    }

    return rollout_td, timing


def train_mappo_floris_multi_env(
    config_path: str,
    *,
    cfg: Optional[MAPPOTorchRLConfig] = None,
    resume_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 10,
) -> None:
    """Train a MAPPO agent on the FLORIS TorchRL environment.

    Args:
        config_path: Path to the FLORIS YAML configuration.
        cfg: Optional configuration object.
        resume_path: Path to a checkpoint .pt file to resume from. If None,
            training starts from scratch at iteration 0.
        checkpoint_dir: Directory to write checkpoints to.
        checkpoint_every: Save a checkpoint every N iterations (and at the end).
    """
    if cfg is None:
        cfg = MAPPOTorchRLConfig()

    device = cfg.device
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _make_single_env() -> FlorisMultiAgentTorchRLEnv:
        return make_floris_torchrl_env(
            config_path=config_path,
            max_steps=cfg.max_steps_per_episode,
            device="cpu",
        )

    env = ParallelEnv(cfg.num_envs, _make_single_env)
    try:
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

        if cfg.minibatch_size > frames_per_batch:
            import warnings
            warnings.warn(
                f"minibatch_size ({cfg.minibatch_size}) > frames_per_batch ({frames_per_batch}); "
                f"clamping minibatch_size to {frames_per_batch}."
            )
            cfg.minibatch_size = frames_per_batch

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.minibatch_size,
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

        # --- Resume logic ---
        start_iteration = 0
        if resume_path is not None:
            start_iteration = load_checkpoint(resume_path, loss_module, optim, device)
            print(f"[MAPPO] Resumed from {resume_path} at iteration {start_iteration}")

        if start_iteration >= cfg.n_iters:
            print(
                f"[MAPPO] start_iteration ({start_iteration}) >= cfg.n_iters "
                f"({cfg.n_iters}); nothing to do. Increase n_iters to keep training."
            )
            return

        last_completed_iteration = start_iteration

        try:
            # Keys ClipPPOLoss may expose beyond the three loss terms used for backprop.
        # Presence varies slightly by TorchRL version, so we check rather than assume.
            EXTRA_LOSS_KEYS = ["entropy", "clip_fraction", "ESS", "kl_approx"]

            for iter_idx in range(start_iteration, cfg.n_iters):
                iter_start = time.perf_counter()

                # --- Rollout collection ---
                rollout_start = time.perf_counter()
                tensordict_data, rollout_timing = _collect_rollout(
                    env=env,
                    policy=policy,
                    max_steps=cfg.max_steps_per_episode,
                    device=device,
                )
                rollout_time = time.perf_counter() - rollout_start

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

                with torch.no_grad():
                    gae(
                        tensordict_data,
                        params=loss_module.critic_network_params,
                        target_params=loss_module.target_critic_network_params,
                    )

                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view)

                # --- Optimisation ---
                cumulative_loss = 0.0
                cumulative_critic = 0.0
                cumulative_actor = 0.0
                cumulative_extra = {k: 0.0 for k in EXTRA_LOSS_KEYS}
                extra_key_seen = {k: False for k in EXTRA_LOSS_KEYS}

                optim_start = time.perf_counter()
                num_minibatches = frames_per_batch // cfg.minibatch_size
                for _ in range(cfg.num_epochs):
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

                        cumulative_loss += loss_value.item()
                        cumulative_critic += loss_vals["loss_critic"].item()
                        cumulative_actor += loss_vals["loss_objective"].item()

                        for k in EXTRA_LOSS_KEYS:
                            if k in loss_vals.keys():
                                extra_key_seen[k] = True
                                v = loss_vals[k]
                                cumulative_extra[k] += v.item() if torch.is_tensor(v) else float(v)
                optim_time = time.perf_counter() - optim_start

                total_updates = cfg.num_epochs * num_minibatches
                iter_time = time.perf_counter() - iter_start

                # --- Reward stats ---
                rewards = tensordict_data.get(("next", "agents", "reward"))
                mean_reward = rewards.mean().item()
                reward_var = rewards.var().item()
                reward_std = rewards.std().item()

                mean_baseline_mw = tensordict_data.get(("next", "baseline_power_mw")).mean().item()
                mean_actual_mw = tensordict_data.get(("next", "actual_power_mw")).mean().item()

                log_dict = {
                    "train/iteration": iter_idx + 1,
                    "train/mean_reward": mean_reward,
                    "train/reward_var": reward_var,
                    "train/reward_std": reward_std,
                    "train/total_loss": cumulative_loss / total_updates,
                    "train/critic_loss": cumulative_critic / total_updates,
                    "train/actor_loss": cumulative_actor / total_updates,
                    "train/baseline_power_mw": mean_baseline_mw,
                    "train/actual_power_mw": mean_actual_mw,
                    # Penalty / clipping coefficients — constant unless you later
                    # anneal them, but logged per-iteration so they show up
                    # alongside everything else in the same chart.
                    "train/entropy_coeff": cfg.entropy_eps,
                    "train/clip_epsilon": cfg.clip_epsilon,


                    # Timing
                    "performance/iteration_time_s": iter_time,
                    "performance/rollout_time_s": rollout_time,
                    "performance/optim_time_s": optim_time,
                    "performance/env_step_time_s": rollout_timing["env_step_time_s"],
                    "performance/policy_time_s": rollout_timing["policy_time_s"],
                    "performance/env_step_time_per_step_s": rollout_timing["env_step_time_per_step_s"],
                    "performance/policy_time_per_step_s": rollout_timing["policy_time_per_step_s"],
                    "performance/frames_per_second": frames_per_batch / rollout_time,
                }

                for k in EXTRA_LOSS_KEYS:
                    if extra_key_seen[k]:
                        log_dict[f"train/{k}"] = cumulative_extra[k] / total_updates

                wandb.log(log_dict)

                print(
                    f"[MAPPO] Iteration {iter_idx + 1}/{cfg.n_iters} "
                    f"- Mean Reward: {mean_reward:.4f} (var {reward_var:.4f}) "
                    f"| Loss: {cumulative_loss / total_updates:.4f} "
                    f"| iter {iter_time:.2f}s (rollout {rollout_time:.2f}s / optim {optim_time:.2f}s)"
                )

                last_completed_iteration = iter_idx + 1

                if checkpoint_every and last_completed_iteration % checkpoint_every == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter{last_completed_iteration}.pt")
                    save_checkpoint(ckpt_path, last_completed_iteration, loss_module, optim)
                    save_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), last_completed_iteration, loss_module, optim)
                    print(f"[MAPPO] Saved checkpoint -> {ckpt_path}")
                    
        finally:
            # Always persist the latest state, even on Ctrl-C or an exception,
            # so you never lose more than the current in-progress iteration.
            save_checkpoint(
                os.path.join(checkpoint_dir, "latest.pt"),
                last_completed_iteration,
                loss_module,
                optim,
            )
            print(f"[MAPPO] Final checkpoint saved at iteration {last_completed_iteration}")
    finally:
        env.close()

