"""Tests for FLORIS multi-agent environment"""

import pytest
import numpy as np
from wind_farm_control.environments import FlorisMultiAgentEnv


class TestFlorisMultiAgentEnv:
    """Test suite for FlorisMultiAgentEnv"""

    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return FlorisMultiAgentEnv("data_generation/farm_types/gch.yaml")

    def test_initialization(self, env):
        """Test that environment initializes correctly"""
        assert len(env.possible_agents) == 4
        assert all(f"turbine_{i}" in env.possible_agents for i in range(4))

    def test_reset(self, env):
        """Test environment reset"""
        observations, infos = env.reset()

        assert len(observations) == 4
        assert all(a in observations for a in env.possible_agents)
        assert all(obs.shape == (3,) for obs in observations.values())
        assert all("state" in infos[a] for a in env.possible_agents)

    def test_step(self, env):
        """Test environment step"""
        env.reset()

        # Create random actions
        actions = {a: np.random.uniform(-1, 1, size=(1,)) for a in env.possible_agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        assert len(obs) == 4
        assert len(rewards) == 4
        assert all(isinstance(r, (float, np.floating)) for r in rewards.values())
        assert all(a in terminations for a in env.possible_agents)

    def test_observation_spaces(self, env):
        """Test observation and action spaces"""
        for agent in env.possible_agents:
            obs_space = env.observation_spaces[agent]
            action_space = env.action_spaces[agent]

            assert obs_space.shape == (3,)  # [wind_dir, wind_speed, turbulence]
            assert action_space.shape == (1,)  # [yaw_action]

    def test_state_shape(self, env):
        """Test global state shape"""
        state = env.state()
        assert state.shape == (12,)  # 3 params * 4 turbines

    def test_episode_termination(self, env):
        """Test that episode terminates after max_steps"""
        env.reset()
        actions = {a: np.array([0.0]) for a in env.possible_agents}

        for _ in range(env.max_steps):
            _, _, terminations, _, _ = env.step(actions)

        assert all(terminations.values())
