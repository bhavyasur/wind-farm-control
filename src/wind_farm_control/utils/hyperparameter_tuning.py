"""Hyperparameter tuning utilities using Optuna"""

import torch
import optuna
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env

from wind_farm_control.environments import FlorisMultiAgentEnv
from wind_farm_control.models import Actor, Critic


def run_hyperparameter_tuning(
    config_path,
    n_trials=10,
    training_timesteps=5000,
    eval_steps=100,
    direction="maximize",
):
    """
    Run hyperparameter tuning using Optuna.

    Args:
        config_path: Path to FLORIS configuration file
        n_trials: Number of Optuna trials to run
        training_timesteps: Number of timesteps for each trial
        eval_steps: Number of steps for evaluation
        direction: Optimization direction ('maximize' or 'minimize')

    Returns:
        Optuna study object with results
    """

    def objective(trial):
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Total reward achieved during evaluation
        """
        # 1. Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        entropy_scale = trial.suggest_float("entropy_scale", 0.001, 0.05)
        mini_batches = trial.suggest_categorical("mini_batches", [2, 4, 8])

        # 2. Environment setup
        env = FlorisMultiAgentEnv(config_path)
        env = wrap_env(env, wrapper="pettingzoo")

        # 3. Memory & Models (Shared weights for all turbines)
        memory = RandomMemory(
            memory_size=1000, num_envs=env.num_envs, device=env.device
        )
        memories = {agent_name: memory for agent_name in env.possible_agents}

        # Create models for each agent
        models = {}
        for agent_name in env.possible_agents:
            models[agent_name] = {
                "policy": Actor(
                    env.observation_spaces[agent_name],
                    env.action_spaces[agent_name],
                    env.device,
                ),
                "value": Critic(
                    env.shared_observation_spaces[agent_name],
                    env.action_spaces[agent_name],
                    env.device,
                ),
            }

        # 4. Agent configuration
        cfg = MAPPO_DEFAULT_CONFIG.copy()
        cfg["learning_rate"] = lr
        cfg["entropy_loss_scale"] = entropy_scale
        cfg["mini_batches"] = mini_batches
        # Note: Disable experiment logging for faster tuning
        if "experiment" in cfg:
            cfg["experiment"]["write_interval"] = 0

        agent = MAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=env.device,
            shared_observation_spaces=env.shared_observation_spaces,
        )

        # 5. Train using SequentialTrainer
        from skrl.trainers.torch import SequentialTrainer

        trainer = SequentialTrainer(
            env=env,
            agents=agent,
            cfg={"timesteps": training_timesteps, "headless": True},
        )
        trainer.train()

        # Evaluate for 1 episode
        total_reward = 0
        obs, _ = env.reset()
        for _ in range(eval_steps):
            with torch.no_grad():
                actions = agent.act(obs, timestep=0, timesteps=0)[0]
                obs, rewards, terminated, truncated, _ = env.step(actions)
                total_reward += sum(rewards.values()) / len(rewards)
                if any(terminated.values()):
                    break

        return total_reward

    # Create and run Optuna study
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 50)
    print("Hyperparameter Tuning Complete!")
    print("=" * 50)
    print(f"Best Parameters: {study.best_params}")
    print(f"Best Value: {study.best_value}")
    print("=" * 50 + "\n")

    return study
