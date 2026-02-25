"""Multi-agent FLORIS environment for wind farm control"""

from gymnasium import spaces
import numpy as np
from floris import FlorisModel
from pettingzoo import ParallelEnv


class FlorisMultiAgentEnv(ParallelEnv):
    """
    Multi-agent environment for wind farm control using FLORIS.

    Each turbine is controlled by an independent agent that adjusts
    its yaw angle to optimize the total farm power output.
    """

    def __init__(self, config_path):
        """
        Initialize the FLORIS multi-agent environment.

        Args:
            config_path: Path to FLORIS configuration YAML file
        """
        super().__init__()

        # 1. Initialize the physics model
        self.fmodel = FlorisModel(config_path)
        D = 126.0
        self.x_layout = [0, 0, 6 * D, 6 * D]
        self.y_layout = [0, 3 * D, 0, 3 * D]
        self.fmodel.set(layout_x=self.x_layout, layout_y=self.y_layout)

        # 2. Define agents
        self.possible_agents = [f"turbine_{i}" for i in range(len(self.x_layout))]
        self.agents = self.possible_agents[:]

        # 3. Define observation and action spaces
        obs_low = np.array([260.0, 5.0, 0.03], dtype=np.float32)
        obs_high = np.array([290.0, 15.0, 0.25], dtype=np.float32)
        obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Local view for Actor
        self.observation_spaces = {a: obs_space for a in self.possible_agents}

        # Global view for Critic (Concatenated observations of all 4 turbines)
        self.state_space = spaces.Box(
            low=np.tile(obs_low, len(self.possible_agents)),
            high=np.tile(obs_high, len(self.possible_agents)),
            dtype=np.float32,
        )
        self.shared_observation_spaces = {
            a: self.state_space for a in self.possible_agents
        }

        self.action_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for a in self.possible_agents
        }

        self.max_steps = 100
        self.current_step = 0
        self.wind_state = np.array([275.0, 10.0, 0.06], dtype=np.float32)

    def state(self):
        """
        Return the global state (concatenated observations of all turbines).

        Returns:
            Global state array [3 params * 4 turbines]
        """
        return np.tile(self.wind_state, len(self.possible_agents))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observations: Dictionary of observations for each agent
            infos: Dictionary of info for each agent
        """
        self.current_step = 0
        self.wind_state = np.array([275.0, 10.0, 0.06], dtype=np.float32)

        observations = {a: self.wind_state for a in self.possible_agents}

        infos = {a: {"state": self.state()} for a in self.possible_agents}

        return observations, infos

    def step(self, actions):
        """
        Execute one step of the environment.

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            terminations: Dictionary of termination flags
            truncations: Dictionary of truncation flags
            infos: Dictionary of info for each agent
        """
        self.current_step += 1

        # Random wind drift
        self.wind_state[0] += np.random.normal(0, 0.2)
        self.wind_state[1] += np.random.normal(0, 0.05)
        self.wind_state = np.clip(self.wind_state, [260, 5, 0.03], [290, 15, 0.25])

        # Apply yaws to FLORIS
        yaws = np.array([actions[a][0] for a in self.possible_agents]) * 25.0
        self.fmodel.set(
            wind_directions=[self.wind_state[0]],
            wind_speeds=[self.wind_state[1]],
            turbulence_intensities=[self.wind_state[2]],
            yaw_angles=np.array([yaws]),
        )
        self.fmodel.run()

        # Global Reward: Sum of all turbine power (encourages coordination)
        reward = np.sum(self.fmodel.get_turbine_powers()) / 1e6
        rewards = {a: reward for a in self.possible_agents}

        terminated = self.current_step >= self.max_steps
        terminations = {a: terminated for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}

        observations = {a: self.wind_state for a in self.possible_agents}

        infos = {a: {"state": self.state()} for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos
