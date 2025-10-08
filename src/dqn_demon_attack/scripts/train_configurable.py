"""
Configurable DQN training script for Atari DemonAttack.

Supports loading different configurations from the configs directory.
Includes improved training features like progress tracking and better logging.
"""

import argparse
import importlib
import os
import random
import sys

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch import nn

from dqn_demon_attack.agents import DQN, Replay
from dqn_demon_attack.envs import make_env, RewardConfig
from dqn_demon_attack.utils import CSVLogger


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def evaluate(q, device="cuda", episodes=10):
    """
    Evaluate the Q-network on multiple episodes.

    Args:
        q: The Q-network to evaluate.
        device: Device to run evaluation on.
        episodes: Number of episodes to evaluate.

    Returns:
        Tuple of (mean_return, std_return, mean_length).
    """
    env = make_env()
    q.eval()
    rets, lens = [], []

    for _ in range(episodes):
        s, _ = env.reset()
        s_t = to_tensor(s).to(device)
        done, ep_ret, ep_len = False, 0.0, 0

        while not done:
            with torch.no_grad():
                a = q(s_t).argmax(1).item()
            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            s_t = to_tensor(s2).to(device)
            ep_ret += float(info.get("raw_reward", r))
            ep_len += 1

        rets.append(ep_ret)
        lens.append(ep_len)

    q.train()
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(lens))


def load_config(config_name):
    """
    Load configuration from configs directory.

    Args:
        config_name: Name of the config file (without .py extension).

    Returns:
        Configuration class.
    """
    try:
        module = importlib.import_module(f"configs.{config_name}")

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Config'):
                return attr

        raise AttributeError(f"No Config class found in configs.{config_name}")
    except ImportError as e:
        print(f"Error: Could not import config '{config_name}': {e}")
        print("Available configs: default, improved, quick, advanced")
        sys.exit(1)


def main(cfg):
    """
    Main training function.

    Args:
        cfg: Configuration object with training hyperparameters.
    """
    set_seed(cfg.seed)

    run_dir = os.path.join("runs", cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log = CSVLogger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=[
            "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
            "loss", "q_mean", "epsilon", "replay_size"
        ]
    )

    reward_cfg_kwargs = {"mode": cfg.reward_mode}
    if hasattr(cfg, 'use_life_penalty'):
        reward_cfg_kwargs['use_life_penalty'] = cfg.use_life_penalty
        reward_cfg_kwargs['life_penalty'] = cfg.life_penalty
    if hasattr(cfg, 'use_streak_bonus'):
        reward_cfg_kwargs['use_streak_bonus'] = cfg.use_streak_bonus
        reward_cfg_kwargs['streak_window'] = cfg.streak_window
        reward_cfg_kwargs['streak_bonus'] = cfg.streak_bonus

    reward_cfg = RewardConfig(**reward_cfg_kwargs)

    env = make_env(
        render_mode=None,
        stack=cfg.frame_stack,
        screen_size=cfg.screen_size,
        terminal_on_life_loss=cfg.terminal_on_life_loss,
        reward_cfg=reward_cfg
    )
    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    q = DQN(n_actions).to(cfg.device)
    tgt = DQN(n_actions).to(cfg.device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_size)

    s, _ = env.reset(seed=cfg.seed)
    s_t = to_tensor(s).to(cfg.device)
    ep_return_raw, ep_return_shaped, ep_len, episode = 0.0, 0.0, 0, 0
    loss_val, q_mean = float("nan"), float("nan")

    print(f"Starting training: {cfg.exp_name}")
    print(f"Total steps: {cfg.total_steps:,}")
    print(f"Device: {cfg.device}")
    print(f"Replay size: {cfg.replay_size:,}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print("-" * 60)

    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * (step / cfg.eps_decay_steps))

        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q(s_t).argmax(1).item()

        s2, r, term, trunc, info = env.step(a)
        done = term or trunc
        raw_r = float(info.get("raw_reward", r))
        ep_return_raw += raw_r
        ep_return_shaped += r
        ep_len += 1

        replay.push(s_t.cpu().numpy()[0], a, r, to_tensor(s2).cpu().numpy()[0], done)
        s_t = to_tensor(s2).to(cfg.device)

        if len(replay) >= cfg.warmup:
            s_b, a_b, r_b, s2_b, d_b = replay.sample(cfg.batch_size)
            s_b = torch.tensor(s_b, dtype=torch.float32, device=cfg.device)
            a_b = torch.tensor(a_b, dtype=torch.int64, device=cfg.device)
            r_b = torch.tensor(r_b, dtype=torch.float32, device=cfg.device)
            s2_b = torch.tensor(s2_b, dtype=torch.float32, device=cfg.device)
            d_b = torch.tensor(d_b, dtype=torch.bool, device=cfg.device)

            with torch.no_grad():
                target = tgt(s2_b).max(1).values
                y = r_b + cfg.gamma * (~d_b).float() * target

            qvals = q(s_b).gather(1, a_b[:, None]).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(qvals, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
            opt.step()
            loss_val = float(loss.item())
            q_mean = float(qvals.mean().item())

            if step % cfg.target_update_freq == 0:
                tgt.load_state_dict(q.state_dict())

        if done:
            episode += 1
            log.log(
                step=step,
                episode=episode,
                ep_return_raw=ep_return_raw,
                ep_return_shaped=ep_return_shaped,
                ep_len=ep_len,
                loss=loss_val,
                q_mean=q_mean,
                epsilon=eps,
                replay_size=len(replay)
            )

            if episode % 10 == 0:
                print(f"[Step {step:7d}] Episode {episode:4d} | "
                      f"Return: {ep_return_raw:6.1f} | "
                      f"Len: {ep_len:4d} | "
                      f"ε: {eps:.3f} | "
                      f"Loss: {loss_val:.4f}")

            ep_return_raw, ep_return_shaped, ep_len = 0.0, 0.0, 0
            s, _ = env.reset()
            s_t = to_tensor(s).to(cfg.device)

        if step % cfg.eval_every == 0:
            avg, std, avg_len = evaluate(q, device=cfg.device, episodes=10)
            torch.save(
                {
                    "step": step,
                    "model": q.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": vars(cfg),
                    "eval_mean": avg,
                    "eval_std": std,
                },
                os.path.join(ckpt_dir, f"step_{step}.pt")
            )
            print("=" * 60)
            print(f"[Eval @ {step}] raw return {avg:.1f}±{std:.1f} | len {avg_len:.0f}")
            print("=" * 60)

    log.close()

    torch.save(
        {
            "step": cfg.total_steps,
            "model": q.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": vars(cfg),
        },
        os.path.join(ckpt_dir, f"final.pt")
    )

    print(f"\nTraining complete! Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on DemonAttack with configurable settings")
    parser.add_argument(
        "--config",
        type=str,
        default="improved",
        help="Config name from configs directory (default, improved, quick, advanced)"
    )
    args = parser.parse_args()

    ConfigClass = load_config(args.config)
    config = ConfigClass()

    main(config)
