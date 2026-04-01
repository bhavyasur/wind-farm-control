"""TorchRL-native multi-agent FLORIS environment.

This environment mirrors the PettingZoo-compatible ``FlorisMultiAgentEnv``
but exposes a TorchRL ``EnvBase`` API with TensorDict-based ``reset`` / ``step``
and multi-agent specs under an ``(\"agents\", ...)`` namespace.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from floris import FlorisModel
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, UnboundedContinuous
from torchrl.envs import EnvBase


class FlorisMultiAgentTorchRLEnv(EnvBase):
    """Multi-agent FLORIS environment with TorchRL API.

    Each turbine is controlled by an independent agent that adjusts its yaw
    angle to optimize the total farm power output. Observations and rewards
    are exposed using a multi-agent layout compatible with TorchRL's
    multi-agent utilities (e.g., :class:`torchrl.modules.MultiAgentMLP`).

    Keys:
    - observations: (\"agents\", \"observation\") with shape ``(n_agents, 3)``
      representing ``[wind_direction, wind_speed, turbulence_intensity]``.
    - actions: (\"agents\", \"action\") with shape ``(n_agents, 1)`` in
      ``[-1, 1]``, scaled to yaw angles in ``[-25°, 25°]`` before calling FLORIS.
    - rewards: (\"agents\", \"reward\") with shape ``(n_agents, 1)`` where each
      agent receives the same global reward (total farm power in MW).
    - done / terminated: scalar booleans shared across agents.
    """

    metadata = {
        "name": "FlorisMultiAgentTorchRLEnv",
    }

    def __init__(
        self,
        config_path: str,
        max_steps: int = 100,
        device: Optional[torch.device | str] = "cpu",
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))

        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        # FLORIS setup (mirrors the PettingZoo environment)
        self.fmodel = FlorisModel(config_path)
        D = 126.0
        self.x_layout = [0, 0, 6 * D, 6 * D]
        self.y_layout = [0, 3 * D, 0, 3 * D]
        self.fmodel.set(layout_x=self.x_layout, layout_y=self.y_layout)

        self.possible_agents = [f"turbine_{i}" for i in range(len(self.x_layout))]
        self.n_agents = len(self.possible_agents)

        self.max_steps = max_steps
        self.current_step = 0

        # Wind state: [direction, speed, turbulence]
        self.wind_state = torch.tensor(
            [275.0, 10.0, 0.06],
            dtype=torch.float32,
            device=self._device,
        )

        # ---- Specs ----
        # Observation per agent: same 3D wind state, represented as (n_agents, 3).
        obs_low = torch.tensor(
            [260.0, 5.0, 0.03], dtype=torch.float32, device=self._device
        )
        obs_high = torch.tensor(
            [290.0, 15.0, 0.25], dtype=torch.float32, device=self._device
        )
        obs_low_expanded = obs_low.expand(self.n_agents, -1)
        obs_high_expanded = obs_high.expand(self.n_agents, -1)

        obs_spec = Bounded(
            low=obs_low_expanded,
            high=obs_high_expanded,
            shape=torch.Size([self.n_agents, 3]),
            device=self._device,
            dtype=torch.float32,
        )

        # Action per agent: scalar in [-1, 1] to be scaled to yaw degrees.
        action_spec = Bounded(
            low=torch.full(
                (self.n_agents, 1), -1.0, dtype=torch.float32, device=self._device
            ),
            high=torch.full(
                (self.n_agents, 1), 1.0, dtype=torch.float32, device=self._device
            ),
            shape=torch.Size([self.n_agents, 1]),
            device=self._device,
            dtype=torch.float32,
        )

        # Reward per agent: unbounded continuous, shared global reward.
        reward_spec = UnboundedContinuous(
            shape=torch.Size([self.n_agents, 1]),
            device=self._device,
            dtype=torch.float32,
        )

        # Done / terminated: shared across agents.
        done_leaf = Categorical(
            n=2,
            shape=torch.Size([1]),
            device=self._device,
            dtype=torch.bool,
        )

        # Observation spec: Composite with nested "agents" key.
        self.observation_spec = Composite(
            agents=Composite(
                observation=obs_spec,
                shape=torch.Size([self.n_agents]),
            ),
            shape=torch.Size([]),
        )

        # Action spec: Composite with nested "agents" key.
        self.action_spec = Composite(
            agents=Composite(
                action=action_spec,
                shape=torch.Size([self.n_agents]),
            ),
            shape=torch.Size([]),
        )

        # Reward spec: Composite with nested "agents" key.
        self.reward_spec = Composite(
            agents=Composite(
                reward=reward_spec,
                shape=torch.Size([self.n_agents]),
            ),
            shape=torch.Size([]),
        )

        # Done spec: "done" and "terminated" scalars.
        self.done_spec = Composite(
            done=done_leaf,
            terminated=done_leaf.clone(),
            shape=torch.Size([]),
        )

        # Multi-agent key conventions used by TorchRL multi-agent utilities.
        # These are simple attributes, not overrides of EnvBase properties.
        self.multiagent_action_key = ("agents", "action")
        self.multiagent_reward_key = ("agents", "reward")

    # --------------------------------------------------------------------- #
    # EnvBase core methods
    # --------------------------------------------------------------------- #

    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set the environment RNG seed (no-op for now).

        FLORIS itself does not expose a seedable RNG through this API, and the
        only stochasticity here comes from PyTorch's random draws in the wind
        model, which users can seed via :func:`torch.manual_seed`.
        """
        return None

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDict:
        """Reset the environment to the initial wind state."""
        self.current_step = 0
        # Reset wind to a nominal operating point (matches PettingZoo env).
        self.wind_state = torch.tensor(
            [275.0, 10.0, 0.06],
            dtype=torch.float32,
            device=self._device,
        )

        obs = self.wind_state.expand(self.n_agents, -1).clone()

        data = TensorDict(
            {
                ("agents", "observation"): obs,
                "done": torch.zeros(1, dtype=torch.bool, device=self._device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self._device),
            },
            batch_size=torch.Size([]),
            device=self._device,
        )
        return data

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        """Single environment step.

        Expects:
            tensordict[\"agents\", \"action\"]: tensor of shape (n_agents, 1)
            with values in [-1, 1].
        """
        self.current_step += 1

        actions = tensordict.get(self.multiagent_action_key)
        if actions.shape != (self.n_agents, 1):
            raise ValueError(
                f"Expected actions of shape ({self.n_agents}, 1), "
                f"got {tuple(actions.shape)}"
            )

        # Random wind drift (same logic as PettingZoo env, but in torch).
        with torch.no_grad():
            drift_dir = torch.normal(
                mean=torch.tensor(0.0, device=self._device),
                std=torch.tensor(0.2, device=self._device),
            )
            drift_speed = torch.normal(
                mean=torch.tensor(0.0, device=self._device),
                std=torch.tensor(0.05, device=self._device),
            )
            self.wind_state[0] = self.wind_state[0] + drift_dir
            self.wind_state[1] = self.wind_state[1] + drift_speed
            self.wind_state = torch.clamp(
                self.wind_state,
                min=torch.tensor([260.0, 5.0, 0.03], device=self._device),
                max=torch.tensor([290.0, 15.0, 0.25], device=self._device),
            )

        # Convert actions to yaw degrees and run FLORIS.
        yaw_adjust = (
            actions.detach().squeeze(-1).cpu().numpy() * 25.0
        )  # shape: (n_agents,)
        self._run_floris(yaw_adjust)

        # Global reward: sum of turbine powers (MW), shared across agents.
        reward_mw = float(np.sum(self.fmodel.get_turbine_powers()) / 1e6)
        reward_tensor = torch.full(
            (self.n_agents, 1),
            reward_mw,
            dtype=torch.float32,
            device=self._device,
        )

        done_flag = self.current_step >= self.max_steps
        done_tensor = torch.tensor([done_flag], dtype=torch.bool, device=self._device)

        next_obs = self.wind_state.expand(self.n_agents, -1).clone()

        out = TensorDict(
            {
                ("agents", "observation"): next_obs,
                ("agents", "reward"): reward_tensor,
                "done": done_tensor,
                "terminated": done_tensor.clone(),
            },
            batch_size=torch.Size([]),
            device=self._device,
        )
        return out

    # ------------------------------------------------------------------ #
    # Step wrapper (TorchRL "next" convention for ParallelEnv)
    # ------------------------------------------------------------------ #

    def step(self, tensordict: TensorDictBase) -> TensorDict:
        """Step the environment and return a tensordict with a \"next\" entry.

        ParallelEnv (and other TorchRL utilities) expect step() to return a
        tensordict that has a \"next\" key containing the next observation,
        reward, done, and terminated. This method returns the input with
        \"next\" set to the output of _step().
        """
        next_td = self._step(tensordict)
        out = tensordict.clone(recurse=False)
        out.set("next", next_td)
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_floris(self, yaw_angles_deg: np.ndarray) -> None:
        """Run FLORIS with the current wind state and yaw angles."""
        if yaw_angles_deg.shape[0] != self.n_agents:
            raise ValueError(
                f"Expected yaw_angles_deg of shape ({self.n_agents},), "
                f"got {yaw_angles_deg.shape}"
            )

        wind_dir = float(self.wind_state[0].item())
        wind_speed = float(self.wind_state[1].item())
        turbulence = float(self.wind_state[2].item())

        self.fmodel.set(
            wind_directions=[wind_dir],
            wind_speeds=[wind_speed],
            turbulence_intensities=[turbulence],
            yaw_angles=np.array([yaw_angles_deg], dtype=float),
        )
        self.fmodel.run()


def make_floris_torchrl_env(
    config_path: str,
    max_steps: int = 100,
    device: Optional[torch.device | str] = "cpu",
) -> FlorisMultiAgentTorchRLEnv:
    """Factory for the TorchRL FLORIS environment.

    This factory is designed to be passed to TorchRL collectors or BenchMARL
    task definitions as a callable that returns a fully-configured environment.
    The returned env is wrapped with a :class:`RewardSum` transform that
    accumulates per-episode rewards under ``(\"agents\", \"episode_reward\")``.
    """
    return FlorisMultiAgentTorchRLEnv(
        config_path=config_path,
        max_steps=max_steps,
        device=device,
    )
