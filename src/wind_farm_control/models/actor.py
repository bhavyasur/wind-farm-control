"""Actor network for MAPPO"""

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model


class Actor(GaussianMixin, Model):
    """
    Actor network that outputs actions for each turbine.

    Uses a Gaussian policy for continuous yaw angle control.
    """

    def __init__(self, observation_space, action_space, device, **kwargs):
        """
        Initialize the Actor network.

        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            device: Torch device (cpu or cuda)
            **kwargs: Additional arguments
        """
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, reduction="sum")

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        """
        Compute the action mean and log standard deviation.

        Args:
            inputs: Dictionary containing 'states' key with observations
            role: Role string (unused but required by skrl)

        Returns:
            Tuple of (action_mean, log_std, empty_dict)
        """
        return self.net(inputs["states"]), self.log_std_parameter, {}
