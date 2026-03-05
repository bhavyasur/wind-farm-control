"""Tests for the TorchRL-native FLORIS multi-agent environment."""

import numpy as np
import pytest
import torch

from wind_farm_control.environments import make_floris_torchrl_env


@pytest.fixture(scope="module")
def torchrl_env():
    """Create a small TorchRL FLORIS env for testing."""
    return make_floris_torchrl_env(
        config_path="data_generation/farm_types/gch.yaml",
        max_steps=5,
        device="cpu",
    )


def test_torchrl_has_compat_checker():
    """TorchRL exposes a built-in environment compatibility checker."""
    from torchrl.envs import check_env_specs  # type: ignore[import]

    assert callable(check_env_specs)


def test_specs_shapes_and_types(torchrl_env):
    """Specs for observation, action, reward and done have the expected shapes."""
    base_env = torchrl_env

    obs_spec = base_env.observation_spec["agents", "observation"]
    action_spec = base_env.action_spec["agents", "action"]
    reward_spec = base_env.reward_spec["agents", "reward"]
    done_spec = base_env.done_spec["done"]
    terminated_spec = base_env.done_spec["terminated"]

    assert obs_spec.shape == torch.Size([base_env.n_agents, 3])
    assert action_spec.shape == torch.Size([base_env.n_agents, 1])
    assert reward_spec.shape == torch.Size([base_env.n_agents, 1])

    assert done_spec.shape == torch.Size([1])
    assert terminated_spec.shape == torch.Size([1])
    assert done_spec.dtype is torch.bool
    assert terminated_spec.dtype is torch.bool


def test_reset_outputs_tensordict_with_expected_keys(torchrl_env):
    """reset() returns a TensorDict with the expected keys, shapes and dtypes."""
    base_env = torchrl_env

    td = torchrl_env.reset()

    # Keys exist
    keys = list(td.keys(include_nested=True))
    assert ("agents", "observation") in keys
    assert "done" in keys
    assert "terminated" in keys

    obs = td["agents", "observation"]
    done = td["done"]
    terminated = td["terminated"]

    assert obs.shape == (base_env.n_agents, 3)
    assert obs.dtype is torch.float32
    assert done.shape == (1,)
    assert terminated.shape == (1,)
    assert not bool(done.item())
    assert not bool(terminated.item())


def test_step_updates_observation_and_reward(torchrl_env):
    """A single step with random actions produces next obs and reward with correct shape."""
    base_env = torchrl_env

    td = torchrl_env.reset()

    # Sample random actions from the spec domain
    action_spec = base_env.action_spec["agents", "action"]
    actions = action_spec.sample()
    assert actions.shape == (base_env.n_agents, 1)

    td.set(("agents", "action"), actions)
    td_next = torchrl_env.step(td)

    next_obs = td_next["agents", "observation"]
    reward = td_next["agents", "reward"]
    done = td_next["done"]

    assert next_obs.shape == (base_env.n_agents, 3)
    assert reward.shape == (base_env.n_agents, 1)
    assert done.shape == (1,)

    # All agents should receive the same global reward
    reward_np = reward.cpu().numpy()
    assert np.allclose(reward_np, reward_np[0])


def test_episode_terminates_after_max_steps(torchrl_env):
    """The shared done flag becomes True after max_steps transitions."""
    base_env = torchrl_env

    td = torchrl_env.reset()
    for _ in range(base_env.max_steps):
        actions = base_env.action_spec["agents", "action"].sample()
        td.set(("agents", "action"), actions)
        td = torchrl_env.step(td)

    assert bool(td["done"].item())

