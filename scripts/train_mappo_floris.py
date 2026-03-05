"""Train MAPPO on the FLORIS TorchRL environment.

This script follows the TorchRL multi-agent PPO tutorial:
https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Each wind turbine is treated as an agent, and we run multiple
independent environments in parallel (default: 8).
"""

from __future__ import annotations

import argparse

import torch

from wind_farm_control.training.mappo_torchrl import (
    MAPPOTorchRLConfig,
    train_mappo_floris_multi_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MAPPO on the FLORIS TorchRL environment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_generation/farm_types/gch.yaml",
        help="Path to FLORIS farm config YAML.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel FLORIS environments to run.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=5,
        help="Number of sampling/training iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. "
        "Defaults to CUDA if available, else CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MAPPOTorchRLConfig(
        device=device,
        num_envs=args.num_envs,
        n_iters=args.n_iters,
    )

    train_mappo_floris_multi_env(
        config_path=args.config,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

