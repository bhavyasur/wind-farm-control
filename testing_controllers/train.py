import os
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from floris_env import FlorisEnv

def train_model(algo_name="PPO", timesteps=20000):
    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "models"
    log_dir = current_dir / "logs"
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    env = FlorisEnv()

    if algo_name.upper() == "PPO":
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4, 
            tensorboard_log=str(log_dir)
        )
    
    elif algo_name.upper() == "DDPG":
        # includes noise exploration for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=0.1 * np.ones(n_actions)
        )
        model = DDPG(
            "MlpPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=1, 
            tensorboard_log=str(log_dir)
        )
    else:
        raise ValueError("choose 'PPO' or 'DDPG'")

    # TRAIN
    print(f"--- Starting training for {algo_name} ({timesteps} steps) ---")
    model.learn(total_timesteps=timesteps, tb_log_name=f"{algo_name}_run")
    
    # SAVE
    save_path = model_dir / f"floris_{algo_name.lower()}_model"
    model.save(str(save_path))
    print(f"Model saved to: {save_path}")

    # BRIEF EVALUATION
    print(f"\nEvaluating {algo_name} briefly:")
    obs, _ = env.reset()
    for i in range(3):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Test {i+1}: Power = {info['power_kw']:.2f} kW | Yaw Angles: {action * 25.0}")
        obs, _ = env.reset()

if __name__ == "__main__":
    # change "PPO" to "DDPG" depending on which model
    train_model(algo_name="DDPG", timesteps=20000)