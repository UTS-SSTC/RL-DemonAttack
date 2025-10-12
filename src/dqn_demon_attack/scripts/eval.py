"""
Evaluation script for trained DQN models.

Loads a checkpoint and evaluates the agent's performance over multiple
episodes, reporting statistics including return, episode length, and Q-values.
"""

import argparse

import numpy as np
import torch
from gymnasium.spaces import Discrete

from dqn_demon_attack.envs import make_env
from dqn_demon_attack.utils.training_utils import load_model, to_tensor


def evaluate(ckpt_path, episodes=10, device="cuda"):
    """
    Evaluate a trained model over multiple episodes.

    Args:
        ckpt_path: Path to the checkpoint file.
        episodes: Number of episodes to evaluate.
        device: Device to run evaluation on ('cuda' or 'cpu').
    """
    env = make_env()
    assert isinstance(env.action_space, Discrete)

    q, _ = load_model(ckpt_path, device)
    q.eval()

    rets, lens, q_means, q_stds = [], [], [], []

    for _ in range(episodes):
        s, _ = env.reset()
        s_t = to_tensor(s).to(device)
        done, ep_ret, ep_len = False, 0.0, 0

        while not done:
            with torch.no_grad():
                q_vals = q(s_t)
                q_mean = q_vals.mean().item()
                q_std = q_vals.std().item()
                a = q_vals.argmax(1).item()

            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            s_t = to_tensor(s2).to(device)
            ep_ret += float(info.get("raw_reward", r))
            ep_len += 1
            q_means.append(q_mean)
            q_stds.append(q_std)

        rets.append(ep_ret)
        lens.append(ep_len)

    mean_ret, std_ret = np.mean(rets), np.std(rets)
    mean_len = np.mean(lens)
    mean_q, std_q = np.mean(q_means), np.mean(q_stds)

    print(f"[Eval] raw return {mean_ret:.1f}±{std_ret:.1f} | len {mean_len:.0f}")
    print(f"        mean Q {mean_q:.3f} | Q std {std_q:.3f}")
    print(f"        best/avg/worst episode = {max(rets):.1f}/{mean_ret:.1f}/{min(rets):.1f}")
    print(f"        score per frame ≈ {mean_ret / mean_len:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"
    evaluate(args.ckpt, args.episodes, device=device)
