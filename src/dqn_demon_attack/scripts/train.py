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

    if cfg.log_file is None:
        cfg.log_file = os.path.join(run_dir, "train.log")

    log = CSVLogger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=[
            "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
            "loss", "q_mean", "epsilon", "replay_size", "lr"
        ]
    )

    log_fh = None
    if cfg.log_file:
        os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
        log_fh = open(cfg.log_file, 'w', encoding='utf-8')

    def log_print(message: str):
        """Print to console and optionally to log file."""
        print(message)
        if log_fh:
            log_fh.write(message + '\n')
            log_fh.flush()

    log_print(f"Starting training: {cfg.exp_name}")
    log_print(f"Model: {cfg.model_type}")
    log_print(f"Total steps: {cfg.total_steps:,}")
    log_print(f"Device: {cfg.device}")
    log_print(f"Prioritized Replay: {cfg.use_prioritized_replay}")
    log_print(f"Double DQN: {cfg.use_double_dqn}")
    if cfg.record_breakthrough:
        log_print(f"Breakthrough Recording: enabled (threshold={cfg.breakthrough_threshold})")
    log_print("-" * 60)

    def on_episode_callback(episode: int, step: int, ep_return_raw: float,
                           ep_return_shaped: float, ep_len: int):
        if episode % 10 == 0:
            msg = (f"[Step {step:7d}] Episode {episode:4d} | "
                   f"Return: {ep_return_raw:6.1f} | "
                   f"Len: {ep_len:4d}")
            log_print(msg)

    def on_checkpoint_callback(step: int, ckpt_path: str):
        log_print("=" * 60)
        log_print(f"[Checkpoint] Saved at step {step}")
        log_print("=" * 60)

    try:
        train_dqn(
            cfg=cfg,
            log=log,
            run_dir=run_dir,
            on_episode_callback=on_episode_callback,
            on_checkpoint_callback=on_checkpoint_callback
        )
    finally:
        if log_fh:
            log_fh.close()

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
    parser.add_argument(
        "--record-breakthrough",
        action="store_true",
        help="Enable breakthrough video recording"
    )
    parser.add_argument(
        "--breakthrough-threshold",
        type=float,
        default=None,
        help="Performance improvement threshold for breakthrough detection (default: 0.15)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of breakthrough videos to keep (default: 10)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file for training output (default: runs/<exp_name>/train.log)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.record_breakthrough:
        config.record_breakthrough = True
    if args.breakthrough_threshold is not None:
        config.breakthrough_threshold = args.breakthrough_threshold
    if args.max_videos is not None:
        config.max_videos = args.max_videos
    if args.log_file is not None:
        config.log_file = args.log_file

    train_with_config(config)


if __name__ == "__main__":
    main()
