import argparse
import torch

from envs import make_env
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


def load_model(ckpt_path, n_actions):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    q = DQN(n_actions)
    q.load_state_dict(ckpt["model"])
    return q


def evaluate(ckpt_path, episodes=10, device="cuda"):
    env = make_env()
    n_actions = env.action_space.n
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

    import numpy as np
    mean_ret, std_ret = np.mean(rets), np.std(rets)
    mean_len = np.mean(lens)
    mean_q, std_q = np.mean(q_means), np.mean(q_stds)

    print(f"[Eval] raw return {mean_ret:.1f}±{std_ret:.1f} | len {mean_len:.0f}")
    print(f"        mean Q {mean_q:.3f} | Q std {std_q:.3f}")
    print(f"        best/avg/worst episode = {max(rets):.1f}/{mean_ret:.1f}/{min(rets):.1f}")
    print(f"        score per frame ≈ {mean_ret / mean_len:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps"
    evaluate(args.ckpt, args.episodes, device=device)
