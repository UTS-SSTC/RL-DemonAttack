"""
Evaluation script for trained DQN models.

Loads a checkpoint and evaluates the agent's performance over multiple
episodes, reporting statistics including return, episode length, and Q-values.
"""

import argparse

import numpy as np
import torch
from gymnasium.spaces import Discrete

from dqn_demon_attack.agents import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN
from dqn_demon_attack.envs import make_env


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


def load_model(ckpt_path, n_actions):
    """
    Load a trained DQN model from checkpoint.

    Automatically detects model architecture based on checkpoint keys.

    Args:
        ckpt_path: Path to the checkpoint file.
        n_actions: Number of actions in the environment.

    Returns:
        Loaded DQN model.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    # Detect model architecture from state dict keys
    has_adv_val = any(k.startswith("adv.") or k.startswith("val.") for k in state_dict.keys())
    has_noisy = any("_sigma" in k or "_epsilon" in k for k in state_dict.keys())

    if has_adv_val and has_noisy:
        q = NoisyDuelingDQN(n_actions)
    elif has_adv_val:
        q = DuelingDQN(n_actions)
    elif has_noisy:
        q = NoisyDQN(n_actions)
    else:
        q = DQN(n_actions)

    q.load_state_dict(state_dict)
    return q


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
    n_actions = int(env.action_space.n)

    q = load_model(ckpt_path, n_actions).to(device)
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
