"""
Visualization script for trained DQN agents.

Supports both human viewing (rendering to screen) and video recording
of agent gameplay for analysis and presentation.
"""

import argparse
import os

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    ResizeObservation,
    RecordEpisodeStatistics,
    RecordVideo
)

from dqn_demon_attack.agents import DQN
from dqn_demon_attack.envs import RewardConfig, DemonAttackReward


def to_tensor(obs):
    """
    Convert observation to normalized PyTorch tensor.

    Handles various observation shapes and converts to [1, C, H, W] format
    with values normalized to [0, 1].

    Args:
        obs: Numpy array observation from environment.

    Returns:
        PyTorch tensor of shape [1, C, H, W] with float32 values in [0, 1].
    """
    arr = np.asarray(obs)

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            pass
        elif arr.shape[1] in (1, 3, 4):
            arr = np.moveaxis(arr, 1, 0)
        elif arr.shape[2] in (1, 3, 4):
            arr = np.moveaxis(arr, 2, 0)
        else:
            raise RuntimeError(f"Unrecognized obs shape {arr.shape}, cannot locate channel axis.")
    else:
        raise RuntimeError(f"Unrecognized obs ndim {arr.ndim}")

    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr[None, ...])


def load_model(ckpt_path, n_actions, device):
    """
    Load a trained DQN model from checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file.
        n_actions: Number of actions in the environment.
        device: Device to load the model on.

    Returns:
        Loaded DQN model in evaluation mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    q = DQN(n_actions).to(device)
    q.load_state_dict(ckpt["model"])
    q.eval()
    return q


def make_human_env(terminal_on_life_loss=True):
    """
    Create environment for human viewing.

    Args:
        terminal_on_life_loss: Whether to end episode on life loss.

    Returns:
        Wrapped environment with human rendering enabled.
    """
    gym.register_envs(ale_py)
    base = gym.make(
        'ALE/DemonAttack-v5',
        render_mode="human",
        frameskip=1,
        repeat_action_probability=0.0
    )
    env = RecordEpisodeStatistics(base)
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
        screen_size=84
    )
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env = DemonAttackReward(env, RewardConfig(mode="clip"))
    return env


def make_video_env(video_folder, terminal_on_life_loss=True):
    """
    Create environment for video recording.

    Args:
        video_folder: Directory to save recorded videos.
        terminal_on_life_loss: Whether to end episode on life loss.

    Returns:
        Wrapped environment with video recording enabled.
    """
    gym.register_envs(ale_py)
    base = gym.make(
        'ALE/DemonAttack-v5',
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.0
    )
    env = RecordEpisodeStatistics(base)
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
        screen_size=84
    )
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env = DemonAttackReward(env, RewardConfig(mode="clip"))

    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix="watch"
    )
    return env


def run_episode(env, q, device):
    """
    Run a single episode with the trained agent.

    Args:
        env: The environment to run in.
        q: The Q-network to use for action selection.
        device: Device the model is on.

    Returns:
        Tuple of (episode_return, episode_length).
    """
    s, _ = env.reset()
    done, ep_ret_raw, ep_len = False, 0.0, 0

    while not done:
        s_t = to_tensor(s).to(device)
        with torch.no_grad():
            a = q(s_t).argmax(1).item()
        s, r, term, trunc, info = env.step(a)
        done = term or trunc
        ep_ret_raw += float(info.get("raw_reward", r))
        ep_len += 1

    return ep_ret_raw, ep_len


def main():
    """Main function for watching trained agent gameplay."""
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "video"],
        help="Display mode: human window or video recording"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--video_folder", type=str, default="runs/exp1/videos", help="Video output folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"

    if args.mode == "human":
        env = make_human_env()
    else:
        env = make_video_env(args.video_folder)

    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    q = load_model(args.ckpt, n_actions, device=device)
    scores = []

    for ep in range(args.episodes):
        ret, length = run_episode(env, q, device)
        print(f"[Episode {ep + 1}] raw return={ret:.1f}, steps={length}")
        scores.append(ret)

    if args.mode == "video":
        print(f"Videos saved to: {os.path.abspath(args.video_folder)}")
    else:
        print("Close the game window to end the program.")


if __name__ == "__main__":
    main()
