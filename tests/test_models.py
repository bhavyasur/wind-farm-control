"""Tests for Actor and Critic models"""

import pytest
import torch
from gymnasium import spaces
import numpy as np
from wind_farm_control.models import Actor, Critic


class TestActor:
    """Test suite for Actor model"""

    @pytest.fixture
    def actor(self):
        """Create a test actor"""
        obs_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        device = torch.device("cpu")
        return Actor(obs_space, action_space, device)

    def test_initialization(self, actor):
        """Test actor initialization"""
        assert actor.num_observations == 3
        assert actor.num_actions == 1

    def test_forward_pass(self, actor):
        """Test actor forward pass"""
        batch_size = 4
        inputs = {"states": torch.randn(batch_size, 3)}

        mean, log_std, info = actor.compute(inputs, role="policy")

        assert mean.shape == (batch_size, 1)
        assert log_std.shape == (1,)
        assert isinstance(info, dict)

    def test_output_range(self, actor):
        """Test that actor outputs are reasonable"""
        inputs = {"states": torch.randn(10, 3)}
        mean, _, _ = actor.compute(inputs, role="policy")

        # Mean should be finite
        assert torch.isfinite(mean).all()


class TestCritic:
    """Test suite for Critic model"""

    @pytest.fixture
    def critic(self):
        """Create a test critic"""
        # Critic uses global state (12 dimensions for 4 turbines)
        obs_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        device = torch.device("cpu")
        return Critic(obs_space, action_space, device)

    def test_initialization(self, critic):
        """Test critic initialization"""
        assert critic.num_observations == 12

    def test_forward_pass(self, critic):
        """Test critic forward pass"""
        batch_size = 4
        inputs = {"states": torch.randn(batch_size, 12)}

        value, info = critic.compute(inputs, role="value")

        assert value.shape == (batch_size, 1)
        assert isinstance(info, dict)

    def test_output_range(self, critic):
        """Test that critic outputs are reasonable"""
        inputs = {"states": torch.randn(10, 12)}
        value, _ = critic.compute(inputs, role="value")

        # Value should be finite
        assert torch.isfinite(value).all()
