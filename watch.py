import argparse
import os
import torch

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, ResizeObservation, RecordEpisodeStatistics, \
    RecordVideo

from envs import RewardConfig, DemonAttackReward
from models import DQN


def to_tensor(obs):
    """
    Convert various [H,W,stack] / [H,stack,W] / [stack,H,W] / [H,W] to [1,C,H,W] float32[0,1]
    """
    import numpy as np, torch
    arr = np.asarray(obs)
    if arr.ndim == 2:
        # Single frame grayscale => [H,W] -> [1,H,W]
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # Find the channel axis (typically 1/3/4)
        if arr.shape[0] in (1, 3, 4):
            # [C,H,W], no need to move
            pass
        elif arr.shape[1] in (1, 3, 4):
            # [H,C,W] -> [C,H,W]
            arr = np.moveaxis(arr, 1, 0)
        elif arr.shape[2] in (1, 3, 4):
            # [H,W,C] -> [C,H,W]
            arr = np.moveaxis(arr, 2, 0)
        else:
            raise RuntimeError(f"Unrecognized obs shape {arr.shape}, cannot locate channel axis.")
    else:
        raise RuntimeError(f"Unrecognized obs ndim {arr.ndim}")

    # Normalization + batch dimension
    arr = arr.astype(np.float32) / 255.0
    # [1,C,H,W]
    return torch.from_numpy(arr[None, ...])


def load_model(ckpt_path, n_actions, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    q = DQN(n_actions).to(device)
    q.load_state_dict(ckpt["model"])
    q.eval()
    return q


def make_human_env(terminal_on_life_loss=True):
    base = gym.make('ALE/DemonAttack-v5', render_mode="human")
    env = RecordEpisodeStatistics(base)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4,
                             scale_obs=False, terminal_on_life_loss=terminal_on_life_loss,
                             screen_size=84)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env = DemonAttackReward(env, RewardConfig(mode="clip"))
    return env


def make_video_env(video_folder, terminal_on_life_loss=True):
    base = gym.make('ALE/DemonAttack-v5', render_mode="rgb_array",
                    frameskip=1,
                    repeat_action_probability=0.0)
    env = RecordEpisodeStatistics(base)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4,
                             scale_obs=False, terminal_on_life_loss=terminal_on_life_loss,
                             screen_size=84)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env = DemonAttackReward(env, RewardConfig(mode="clip"))
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True, name_prefix="watch")
    return env


def run_episode(env, q, device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pt")
    parser.add_argument("--mode", type=str, default="human", choices=["human", "video"],
                        help="human window or video recording")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--video_folder", type=str, default="runs/exp1/videos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"

    # Build env (also gives us action space)
    if args.mode == "human":
        env = make_human_env()
    else:
        env = make_video_env(args.video_folder)
    n_actions = env.action_space.n

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
