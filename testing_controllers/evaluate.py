import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO, DDPG
from floris_env import FlorisEnv

def evaluate_models(n_episodes=50):
    # PATHS
    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "models"
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    env = FlorisEnv()
    
    # LOAD MODELS
    models = {}
    if (model_dir / "floris_ppo_model.zip").exists():
        models["PPO"] = PPO.load(model_dir / "floris_ppo_model", env=env)
    if (model_dir / "floris_ddpg_model.zip").exists():
        models["DDPG"] = DDPG.load(model_dir / "floris_ddpg_model", env=env)

    if not models:
        print("No trained models found in /models. Please run train.py first.")
        return

    # EVALUATION LOOP
    results = []

    print(f"Evaluating models over {n_episodes} random wind conditions...")
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        
        # Baseline
        zero_action = np.zeros(env.action_space.shape)
        _, baseline_reward, _, _, info = env.step(zero_action)
        baseline_power = info['power_kw']

        episode_data = {"Episode": i, "Baseline_kW": baseline_power}

        # RL Power Prediction
        for name, model in models.items():
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, info = env.step(action)
            episode_data[f"{name}_kW"] = info['power_kw']
            episode_data[f"{name}_Gain_%"] = 100 * (info['power_kw'] - baseline_power) / baseline_power

        results.append(episode_data)

    # DATA ANALYSIS
    df = pd.DataFrame(results)
    summary = df.mean(numeric_only=True)
    
    print("\n--- Mean Results ---")
    print(f"Average Baseline Power: {summary['Baseline_kW']:.2f} kW")
    for name in models.keys():
        print(f"{name} Average Power: {summary[f'{name}_kW']:.2f} kW ({summary[f'{name}_Gain_%']:.2f}% Gain)")

    # VISUALIZATION
    plt.figure(figsize=(10, 6))
    comparison_metrics = [f"{name}_Gain_%" for name in models.keys()]
    df[comparison_metrics].boxplot()
    plt.title(f"Power Gain % over Baseline (n={n_episodes} random conditions)")
    plt.ylabel("Gain (%)")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(results_dir / "model_comparison_boxplot.png")
    
    df.to_csv(results_dir / "evaluation_results.csv", index=False)
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    evaluate_models(n_episodes=50)