"""
Metrics visualization utilities for training curves.

Provides functions to generate training curve plots from CSV logs.
"""

import csv
import os
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_training_log(log_path: str) -> Dict[str, List]:
    """
    Read training log CSV file.

    Args:
        log_path: Path to the CSV log file.

    Returns:
        Dictionary mapping column names to lists of values.
    """
    data = {}

    if not os.path.exists(log_path):
        return data

    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    if key in ['step', 'episode', 'ep_len', 'replay_size']:
                        data[key].append(int(value))
                    else:
                        data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(value)

    return data


def plot_training_curves(
    log_path: str,
    output_path: str,
    smooth_window: int = 10
) -> bool:
    """
    Generate training curves visualization.

    Creates a multi-panel plot showing episode returns, loss, Q-values,
    and epsilon over training steps.

    Args:
        log_path: Path to the training log CSV file.
        output_path: Path to save the generated plot image.
        smooth_window: Window size for smoothing curves.

    Returns:
        True if successful, False otherwise.
    """
    data = read_training_log(log_path)

    if not data or 'step' not in data:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    def smooth(values, window):
        """Apply moving average smoothing."""
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return smoothed

    steps = data.get('step', [])

    if 'ep_return_raw' in data:
        ax = axes[0, 0]
        returns = data['ep_return_raw']
        returns_smooth = smooth(returns, smooth_window)

        ax.plot(steps, returns, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
        ax.plot(steps, returns_smooth, color='blue', linewidth=2, label='Smoothed')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Episode Return')
        ax.set_title('Episode Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'loss' in data:
        ax = axes[0, 1]
        loss = [v for v in data['loss'] if not np.isnan(v)]
        loss_steps = steps[:len(loss)]
        loss_smooth = smooth(loss, smooth_window)

        ax.plot(loss_steps, loss, alpha=0.3, color='red', linewidth=0.5, label='Raw')
        ax.plot(loss_steps, loss_smooth, color='red', linewidth=2, label='Smoothed')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'q_mean' in data:
        ax = axes[1, 0]
        q_mean = [v for v in data['q_mean'] if not np.isnan(v)]
        q_steps = steps[:len(q_mean)]
        q_smooth = smooth(q_mean, smooth_window)

        ax.plot(q_steps, q_mean, alpha=0.3, color='green', linewidth=0.5, label='Raw')
        ax.plot(q_steps, q_smooth, color='green', linewidth=2, label='Smoothed')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Q-value')
        ax.set_title('Mean Q-value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'epsilon' in data:
        ax = axes[1, 1]
        epsilon = data['epsilon']

        ax.plot(steps, epsilon, color='purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return True


def get_training_summary(log_path: str) -> Optional[Dict]:
    """
    Extract summary statistics from training log.

    Args:
        log_path: Path to the training log CSV file.

    Returns:
        Dictionary with summary statistics or None if log doesn't exist.
    """
    data = read_training_log(log_path)

    if not data or 'step' not in data:
        return None

    returns = [v for v in data.get('ep_return_raw', []) if not np.isnan(v)]
    losses = [v for v in data.get('loss', []) if not np.isnan(v)]
    q_values = [v for v in data.get('q_mean', []) if not np.isnan(v)]

    summary = {
        'total_steps': data['step'][-1] if data['step'] else 0,
        'total_episodes': data['episode'][-1] if data.get('episode') else 0,
    }

    if returns:
        summary['mean_return'] = float(np.mean(returns))
        summary['std_return'] = float(np.std(returns))
        summary['max_return'] = float(np.max(returns))
        summary['min_return'] = float(np.min(returns))

        last_n = min(20, len(returns))
        summary['recent_mean_return'] = float(np.mean(returns[-last_n:]))

    if losses:
        summary['mean_loss'] = float(np.mean(losses))
        summary['final_loss'] = float(losses[-1])

    if q_values:
        summary['mean_q_value'] = float(np.mean(q_values))
        summary['final_q_value'] = float(q_values[-1])

    return summary
