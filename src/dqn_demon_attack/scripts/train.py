"""
YAML-based DQN training script for Atari DemonAttack.

Supports all DQN variants and improvements with YAML configuration.
"""

import argparse
import os

from dqn_demon_attack.utils import CSVLogger, TrainingConfig, load_config, save_config, validate_config
from dqn_demon_attack.utils.training_utils import train_dqn


def train_with_config(cfg: TrainingConfig):
    """Main training function."""
    errors = validate_config(cfg)
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return

    run_dir = os.path.join("runs", cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    log = CSVLogger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=[
            "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
            "loss", "q_mean", "epsilon", "replay_size"
        ]
    )

    print(f"Starting training: {cfg.exp_name}")
    print(f"Model: {cfg.model_type}")
    print(f"Total steps: {cfg.total_steps:,}")
    print(f"Device: {cfg.device}")
    print(f"Prioritized Replay: {cfg.use_prioritized_replay}")
    print(f"Double DQN: {cfg.use_double_dqn}")
    print("-" * 60)

    def on_episode_callback(episode: int, step: int, ep_return_raw: float,
                           ep_return_shaped: float, ep_len: int):
        if episode % 10 == 0:
            print(f"[Step {step:7d}] Episode {episode:4d} | "
                  f"Return: {ep_return_raw:6.1f} | "
                  f"Len: {ep_len:4d}")

    def on_checkpoint_callback(step: int, ckpt_path: str):
        print("=" * 60)
        print(f"[Checkpoint] Saved at step {step}")
        print("=" * 60)

    train_dqn(
        cfg=cfg,
        log=log,
        run_dir=run_dir,
        on_episode_callback=on_episode_callback,
        on_checkpoint_callback=on_checkpoint_callback
    )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    print(f"\nTraining complete! Checkpoints saved to {ckpt_dir}")


def main():
    """Entry point for the CLI command."""
    parser = argparse.ArgumentParser(description="Train DQN with YAML configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_with_config(config)


if __name__ == "__main__":
    main()
