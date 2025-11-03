"""
Evaluation script for trained DQN models.

Loads a checkpoint and evaluates the agent's performance over multiple
episodes, reporting statistics including return, episode length, and Q-values.
"""

import argparse
import os
from typing import Optional, Callable, Dict, Any, List

import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.wrappers import RecordVideo

from dqn_demon_attack.envs import make_env
from dqn_demon_attack.utils.training_utils import load_model, to_tensor


def evaluate(
    ckpt_path: str,
    episodes: int = 10,
    device: str = "cuda",
    render_mode: Optional[str] = None,
    video_folder: Optional[str] = None,
    on_episode_end: Optional[Callable[[int, float, int, Dict[str, float]], None]] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model over multiple episodes.

    Args:
        ckpt_path: Path to the checkpoint file.
        episodes: Number of episodes to evaluate.
        device: Device to run evaluation on ('cuda' or 'cpu').
        render_mode: Render mode for environment ('rgb_array', 'human', or None).
        video_folder: If provided, records videos to this folder.
        on_episode_end: Optional callback called after each episode with
                       (episode_idx, return, length, q_stats).

    Returns:
        Dict containing evaluation results:
            - returns: List of episode returns
            - lengths: List of episode lengths
            - q_means: List of mean Q-values per episode
            - q_stds: List of Q-value standard deviations per episode
            - summary: Dict with aggregate statistics
            - videos: List of video paths (if video_folder is provided)
    """
    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        env = make_env(render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda _: True,
            name_prefix="episode",
            disable_logger=True
        )
    else:
        env = make_env(render_mode=render_mode)

    assert isinstance(env.action_space, Discrete)

    q, _ = load_model(ckpt_path, device)
    q.eval()

    rets, lens, q_means_per_ep, q_stds_per_ep = [], [], [], []
    video_paths: List[str] = []
    existing_videos = set()

    for ep_idx in range(episodes):
        s, _ = env.reset()

        if video_folder is not None and ep_idx > 0:
            current_videos = set(f for f in os.listdir(video_folder) if f.endswith(".mp4"))
            new_videos = current_videos - existing_videos
            if new_videos:
                latest_video = sorted(new_videos)[-1]
                video_paths.append(os.path.join(video_folder, latest_video))
                existing_videos = current_videos

        s_t = to_tensor(s).to(device)
        done, ep_ret, ep_len = False, 0.0, 0
        q_vals_episode = []

        while not done:
            with torch.no_grad():
                q_vals = q(s_t)
                q_mean = q_vals.mean().item()
                q_std = q_vals.std().item()
                a = q_vals.argmax(1).item()
                q_vals_episode.append({"mean": q_mean, "std": q_std})

            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            s_t = to_tensor(s2).to(device)
            ep_ret += float(info.get("raw_reward", r))
            ep_len += 1

        ep_q_mean = np.mean([qv["mean"] for qv in q_vals_episode]) if q_vals_episode else 0.0
        ep_q_std = np.mean([qv["std"] for qv in q_vals_episode]) if q_vals_episode else 0.0

        rets.append(ep_ret)
        lens.append(ep_len)
        q_means_per_ep.append(ep_q_mean)
        q_stds_per_ep.append(ep_q_std)

        if on_episode_end is not None:
            on_episode_end(ep_idx, ep_ret, ep_len, {"q_mean": float(ep_q_mean), "q_std": float(ep_q_std)})

    env.close()

    if video_folder is not None:
        current_videos = set(f for f in os.listdir(video_folder) if f.endswith(".mp4"))
        new_videos = current_videos - existing_videos
        if new_videos:
            latest_video = sorted(new_videos)[-1]
            video_paths.append(os.path.join(video_folder, latest_video))

    mean_ret, std_ret = np.mean(rets), np.std(rets)
    mean_len = np.mean(lens)
    mean_q = np.mean(q_means_per_ep)
    mean_q_std = np.mean(q_stds_per_ep)

    summary = {
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "min_return": float(min(rets)),
        "max_return": float(max(rets)),
        "mean_length": float(mean_len),
        "mean_q": float(mean_q),
        "mean_q_std": float(mean_q_std),
        "score_per_frame": float(mean_ret / mean_len) if mean_len > 0 else 0.0
    }

    return {
        "returns": rets,
        "lengths": lens,
        "q_means": q_means_per_ep,
        "q_stds": q_stds_per_ep,
        "summary": summary,
        "videos": video_paths
    }


def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print evaluation results summary.

    Args:
        results: Results dictionary from evaluate function.
    """
    summary = results["summary"]
    print(f"[Eval] raw return {summary['mean_return']:.1f}±{summary['std_return']:.1f} | "
          f"len {summary['mean_length']:.0f}")
    print(f"        mean Q {summary['mean_q']:.3f} | Q std {summary['mean_q_std']:.3f}")
    print(f"        best/avg/worst episode = {summary['max_return']:.1f}/"
          f"{summary['mean_return']:.1f}/{summary['min_return']:.1f}")
    print(f"        score per frame ≈ {summary['score_per_frame']:.4f}")


def main():
    """
    CLI entry point for evaluation script.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"
    results = evaluate(args.ckpt, args.episodes, device=device)
    print_evaluation_summary(results)


if __name__ == "__main__":
    main()
