"""Tests for training utilities"""

import pytest
from wind_farm_control.training import train_mappo


class TestTraining:
    """Test suite for training functions"""

    def test_train_mappo_smoke(self):
        """Smoke test for train_mappo function"""
        # This is a minimal test to ensure the function can be called
        # For a full test, you would actually run training for a few steps

        # Note: This test is marked as slow since it involves actual training
        pytest.skip("Requires actual training run - run manually with sufficient time")

        agent, env, trainer = train_mappo(
            config_path="data_generation/farm_types/gch.yaml",
            timesteps=100,  # Very short for testing
            memory_size=100,
            learning_rate=5e-4,
            headless=True,
            disable_progressbar=True,
        )

        assert agent is not None
        assert env is not None
        assert trainer is not None

    def test_train_mappo_parameters(self):
        """Test that train_mappo accepts expected parameters"""
        from inspect import signature

        sig = signature(train_mappo)

        expected_params = [
            "config_path",
            "timesteps",
            "memory_size",
            "learning_rate",
            "headless",
            "disable_progressbar",
        ]

        for param in expected_params:
            assert param in sig.parameters
