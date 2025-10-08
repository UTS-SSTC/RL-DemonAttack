"""
Configuration management utilities.

Provides YAML-based configuration loading and validation for training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    All hyperparameters and settings for DQN training on DemonAttack.
    """

    exp_name: str = "exp1"
    total_steps: int = 500_000
    eval_every: int = 50_000
    seed: int = 0
    device: str = "cuda"

    terminal_on_life_loss: bool = True
    screen_size: int = 84
    frame_stack: int = 4

    model_type: str = "DQN"
    reward_mode: str = "clip"

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 32
    replay_size: int = 50_000
    warmup: int = 5_000
    target_update_freq: int = 10_000
    grad_clip: float = 10.0

    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 1_000_000

    use_double_dqn: bool = True
    use_prioritized_replay: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    use_reward_shaping: bool = False
    use_life_penalty: bool = False
    life_penalty: float = -1.0
    use_streak_bonus: bool = False
    streak_window: int = 12
    streak_bonus: float = 0.1


def load_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        TrainingConfig object with loaded parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: TrainingConfig object to save.
        save_path: Path where to save the YAML file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def validate_config(config: TrainingConfig) -> list[str]:
    """
    Validate configuration parameters.

    Args:
        config: TrainingConfig object to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if config.total_steps <= 0:
        errors.append("total_steps must be positive")

    if config.eval_every <= 0:
        errors.append("eval_every must be positive")

    if config.batch_size <= 0:
        errors.append("batch_size must be positive")

    if config.replay_size < config.batch_size:
        errors.append("replay_size must be >= batch_size")

    if config.warmup > config.total_steps:
        errors.append("warmup cannot exceed total_steps")

    if not 0 <= config.gamma <= 1:
        errors.append("gamma must be in [0, 1]")

    if config.lr <= 0:
        errors.append("learning rate must be positive")

    if not 0 <= config.eps_start <= 1:
        errors.append("eps_start must be in [0, 1]")

    if not 0 <= config.eps_end <= 1:
        errors.append("eps_end must be in [0, 1]")

    if config.eps_end > config.eps_start:
        errors.append("eps_end must be <= eps_start")

    if config.model_type not in ["DQN", "DuelingDQN", "NoisyDQN", "NoisyDuelingDQN"]:
        errors.append(f"Invalid model_type: {config.model_type}")

    if config.reward_mode not in ["clip", "scaled", "raw"]:
        errors.append(f"Invalid reward_mode: {config.reward_mode}")

    if config.device not in ["cuda", "cpu", "mps"]:
        errors.append(f"Invalid device: {config.device}")

    if config.use_prioritized_replay:
        if not 0 <= config.per_alpha <= 1:
            errors.append("per_alpha must be in [0, 1]")
        if not 0 <= config.per_beta_start <= 1:
            errors.append("per_beta_start must be in [0, 1]")

    return errors


def create_default_config(save_path: str):
    """
    Create a default configuration file as a template.

    Args:
        save_path: Path where to save the default config.
    """
    config = TrainingConfig()
    save_config(config, save_path)
    print(f"Default configuration saved to: {save_path}")
