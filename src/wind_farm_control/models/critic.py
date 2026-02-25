"""Critic network for MAPPO"""

import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model


class Critic(DeterministicMixin, Model):
    """
    Critic network that estimates the value function.

    Takes the global state (concatenated observations) as input.
    """

    def __init__(self, observation_space, action_space, device, **kwargs):
        """
        Initialize the Critic network.

        Args:
            observation_space: Observation space (should be shared/global state space)
            action_space: Action space from the environment
            device: Torch device (cpu or cuda)
            **kwargs: Additional arguments
        """
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        """
        Compute the value estimate.

        Args:
            inputs: Dictionary containing 'states' key with global state
            role: Role string (unused but required by skrl)

        Returns:
            Tuple of (value_estimate, empty_dict)
        """
        return self.net(inputs["states"]), {}
