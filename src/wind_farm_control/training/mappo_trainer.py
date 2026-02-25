"""MAPPO training utilities"""

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from wind_farm_control.environments import FlorisMultiAgentEnv
from wind_farm_control.models import Actor, Critic


def train_mappo(
    config_path,
    timesteps=50000,
    memory_size=2000,
    learning_rate=5e-4,
    headless=True,
    disable_progressbar=False,
    **trainer_kwargs,
):
    """
    Train a MAPPO agent on the FLORIS multi-agent environment.

    Args:
        config_path: Path to FLORIS configuration file
        timesteps: Number of training timesteps
        memory_size: Size of the replay memory
        learning_rate: Learning rate for the optimizer
        headless: Whether to run without rendering
        disable_progressbar: Whether to disable the progress bar
        **trainer_kwargs: Additional arguments for the trainer config

    Returns:
        Tuple of (agent, env, trainer)
    """
    # Environment setup
    raw_env = FlorisMultiAgentEnv(config_path)
    env = wrap_env(raw_env, wrapper="pettingzoo")

    # Memory object
    memory = RandomMemory(
        memory_size=memory_size, num_envs=env.num_envs, device=env.device
    )

    # Wrapped in a dictionary for MAPPO
    memories = {agent_name: memory for agent_name in env.possible_agents}

    # Model sharing (all agents share the same policy and value networks)
    shared_policy = Actor(
        env.observation_spaces["turbine_0"], env.action_spaces["turbine_0"], env.device
    )

    # Critic uses the shared/global state space
    shared_value = Critic(
        env.shared_observation_spaces["turbine_0"],
        env.action_spaces["turbine_0"],
        env.device,
    )

    models = {
        a: {"policy": shared_policy, "value": shared_value} for a in env.possible_agents
    }

    # MAPPO configuration
    cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
    cfg_agent["random_timesteps"] = 0  # Start learning immediately
    cfg_agent["learning_rate"] = learning_rate
    cfg_agent["state_preprocessor"] = None  # Optional: helps with stability

    # Initialize MAPPO agent
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg_agent,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        shared_observation_spaces=env.shared_observation_spaces,
    )

    # Trainer configuration
    trainer_cfg = {
        "timesteps": timesteps,
        "headless": headless,
        "disable_progressbar": disable_progressbar,
    }
    trainer_cfg.update(trainer_kwargs)

    # Initialize and run trainer
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    trainer.train()

    return agent, env, trainer
