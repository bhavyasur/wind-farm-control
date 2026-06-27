"""Train MAPPO on the FLORIS TorchRL environment.

This script follows the TorchRL multi-agent PPO tutorial:
https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html

Each wind turbine is treated as an agent, and we run multiple
independent environments in parallel (default: 8).
"""

from __future__ import annotations

import argparse
import torch
import wandb  
import os

os.environ["WANDB_START_METHOD"] = "thread"

from mappo_torchrl import (
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
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="torchrl-mappo-floris",
        help="W&B project name.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save/load checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a checkpoint every N iterations.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Total target iteration count. When resuming, set this higher than "
            "the previous total to keep training further.",
    )
    return parser.parse_args()
    


def main() -> None:
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg = MAPPOTorchRLConfig(device=device)
    if args.n_iters is not None:
        cfg.n_iters = args.n_iters

    # Peek at the checkpoint to recover the wandb run id, so resumed runs
    # continue the same wandb chart instead of starting a new one.
    resume_id = None
    if args.resume is not None:
        peek = torch.load(args.resume, map_location="cpu")
        resume_id = peek.get("wandb_run_id")
        del peek

    wandb.init(
        project=args.project,
        config={**vars(args), **cfg.__dict__},
        monitor_gym=True,
        save_code=True,
        mode="online",
        id=resume_id,
        resume="allow" if resume_id else None,
    )

    try:
        train_mappo_floris_multi_env(
            config_path=args.config,
            cfg=cfg,
            resume_path=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()