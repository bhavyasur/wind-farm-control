"""Evaluation utilities for MAPPO agents"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_mappo_vs_baseline(mappo_agent, env, n_episodes=20):
    """
    Evaluate MAPPO agent performance against a greedy baseline.

    The baseline uses zero yaw angles for all turbines.

    Args:
        mappo_agent: Trained MAPPO agent
        env: Wrapped FLORIS environment
        n_episodes: Number of evaluation episodes

    Returns:
        DataFrame with evaluation results
    """
    results = []
    device = mappo_agent.device

    # Set model to evaluation mode (turns off noise/exploration)
    for model_dict in mappo_agent.models.values():
        model_dict["policy"].eval()

    print(f"Starting MAPPO Evaluation over {n_episodes} episodes...")

    for i in range(n_episodes):
        # 1. Reset Environment
        obs_dict, _ = env.reset()

        # 2. Greedy Baseline (All Yaw = 0)
        env.fmodel.set(yaw_angles=np.zeros((1, 4)))
        env.fmodel.run()
        base_power = np.sum(env.fmodel.get_turbine_powers()) / 1e3

        # 3. MAPPO Run
        # Convert numpy observations to torch tensors for skrl
        with torch.no_grad():
            torch_obs = {
                a: torch.as_tensor(obs, device=device, dtype=torch.float32).view(1, -1)
                for a, obs in obs_dict.items()
            }

            # skrl MAPPO act() expects a dict of tensors
            # We use 'act' to get the actions based on current policy
            actions, _, _ = mappo_agent.act(torch_obs, timestep=0, timesteps=0)

            # Extract values from tensors back to numpy for FLORIS
            yaws = (
                np.array(
                    [actions[f"turbine_{j}"].cpu().numpy().flatten() for j in range(4)]
                ).flatten()
                * 25.0
            )

        # 4. Run FLORIS with MAPPO yaws
        env.fmodel.set(yaw_angles=np.array([yaws]))
        env.fmodel.run()
        mappo_power = np.sum(env.fmodel.get_turbine_powers()) / 1e3

        results.append(
            {
                "Episode": i,
                "Base_kW": base_power,
                "MAPPO_kW": mappo_power,
                "Gain_%": 100 * (mappo_power - base_power) / base_power,
            }
        )

    print("Evaluation Complete.")
    return pd.DataFrame(results)


def plot_mappo_performance(df_results):
    """
    Plot MAPPO performance results.

    Creates two plots:
    1. Box plot of power gains distribution
    2. Episode-by-episode comparison of baseline vs MAPPO

    Args:
        df_results: DataFrame from evaluate_mappo_vs_baseline
    """
    # Set the style for a clean, academic look
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Statistical Box Plot (The Gain Distribution) ---
    sns.boxplot(y=df_results["Gain_%"], ax=axes[0], color="#5da5da", width=0.4)
    sns.stripplot(y=df_results["Gain_%"], ax=axes[0], color="black", alpha=0.3)

    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("Distribution of Power Gains (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Gain over Greedy Baseline (%)")

    # --- 2. Episode Comparison (Baseline vs MAPPO) ---
    axes[1].plot(
        df_results["Episode"],
        df_results["Base_kW"],
        label="Baseline (0° Yaw)",
        marker="o",
        linestyle="-",
        color="gray",
        alpha=0.6,
    )
    axes[1].plot(
        df_results["Episode"],
        df_results["MAPPO_kW"],
        label="MAPPO Strategy",
        marker="s",
        linestyle="-",
        color="#ee6677",
    )

    axes[1].set_title("Power Output Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Farm Power (kW)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
