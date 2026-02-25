"""
Example script demonstrating how to use the wind farm control package.

This script shows:
1. Training a MAPPO agent
2. Evaluating performance
3. Saving/loading models
"""

from wind_farm_control.training import train_mappo
from wind_farm_control.evaluation import (
    evaluate_mappo_vs_baseline,
    plot_mappo_performance,
)


def main():
    """Main training and evaluation pipeline"""

    print("=" * 60)
    print("Wind Farm Control - MAPPO Training Example")
    print("=" * 60)

    # Configuration
    config_path = "data_generation/farm_types/gch.yaml"

    # Step 1: Train the agent
    print("\n[1/2] Training MAPPO agent...")
    agent, env, trainer = train_mappo(
        config_path=config_path,
        timesteps=10000,  # Short training for demo
        memory_size=2000,
        learning_rate=5e-4,
        headless=True,
        disable_progressbar=False,
    )
    print("✓ Training complete!")

    # Step 2: Evaluate performance
    print("\n[2/2] Evaluating performance...")
    results = evaluate_mappo_vs_baseline(agent, env, n_episodes=10)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(results["Gain_%"].describe())
    print(f"\nMean power gain: {results['Gain_%'].mean():.2f}%")
    print(f"Best episode gain: {results['Gain_%'].max():.2f}%")
    print("=" * 60)

    # Plot results
    print("\nGenerating plots...")
    plot_mappo_performance(results)

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
