"""
YAML-based DQN training script for Atari DemonAttack.

Supports all DQN variants and improvements with YAML configuration.
"""

import argparse
import os
import random

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch import nn

from dqn_demon_attack.agents import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN, Replay, PrioritizedReplay
from dqn_demon_attack.envs import make_env, RewardConfig
from dqn_demon_attack.utils import CSVLogger, TrainingConfig, load_config, save_config, validate_config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def to_tensor(obs):
    """Convert observation to normalized PyTorch tensor."""
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
            raise RuntimeError(f"Unrecognized obs shape {arr.shape}")
    else:
        raise RuntimeError(f"Unrecognized obs ndim {arr.ndim}")

    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr[None, ...])


def create_model(model_type: str, n_actions: int):
    """Create model based on configuration."""
    model_map = {
        "DQN": DQN,
        "DuelingDQN": DuelingDQN,
        "NoisyDQN": NoisyDQN,
        "NoisyDuelingDQN": NoisyDuelingDQN,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_map[model_type](n_actions)


def evaluate(q, device="cuda", episodes=10):
    """Evaluate the Q-network on multiple episodes."""
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


def main(cfg: TrainingConfig):
    """Main training function."""
    errors = validate_config(cfg)
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return

    set_seed(cfg.seed)

    run_dir = os.path.join("runs", cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    log = CSVLogger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=[
            "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
            "loss", "q_mean", "epsilon", "replay_size"
        ]
    )

    reward_cfg_kwargs: dict[str, str | int | float | bool] = {"mode": cfg.reward_mode}
    if cfg.use_reward_shaping:
        if cfg.use_life_penalty:
            reward_cfg_kwargs['use_life_penalty'] = True
            reward_cfg_kwargs['life_penalty'] = cfg.life_penalty
        if cfg.use_streak_bonus:
            reward_cfg_kwargs['use_streak_bonus'] = True
            reward_cfg_kwargs['streak_window'] = cfg.streak_window
            reward_cfg_kwargs['streak_bonus'] = cfg.streak_bonus

    reward_cfg = RewardConfig(**reward_cfg_kwargs)  # type: ignore[arg-type]

    env = make_env(
        render_mode=None,
        stack=cfg.frame_stack,
        screen_size=cfg.screen_size,
        terminal_on_life_loss=cfg.terminal_on_life_loss,
        reward_cfg=reward_cfg
    )
    assert isinstance(env.action_space, Discrete)
    n_actions = int(env.action_space.n)

    q = create_model(cfg.model_type, n_actions).to(cfg.device)
    tgt = create_model(cfg.model_type, n_actions).to(cfg.device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    if cfg.use_prioritized_replay:
        replay = PrioritizedReplay(
            cap=cfg.replay_size,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            beta_frames=cfg.per_beta_frames
        )
    else:
        replay = Replay(cfg.replay_size)

    s, _ = env.reset(seed=cfg.seed)
    s_t = to_tensor(s).to(cfg.device)
    ep_return_raw, ep_return_shaped, ep_len, episode = 0.0, 0.0, 0, 0
    loss_val, q_mean = float("nan"), float("nan")

    is_noisy = "Noisy" in cfg.model_type

    print(f"Starting training: {cfg.exp_name}")
    print(f"Model: {cfg.model_type}")
    print(f"Total steps: {cfg.total_steps:,}")
    print(f"Device: {cfg.device}")
    print(f"Prioritized Replay: {cfg.use_prioritized_replay}")
    print(f"Double DQN: {cfg.use_double_dqn}")
    print("-" * 60)

    for step in range(1, cfg.total_steps + 1):
        if is_noisy:
            q.reset_noise()
            tgt.reset_noise()

        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * (step / cfg.eps_decay_steps))

        if random.random() < eps and not is_noisy:
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
            if cfg.use_prioritized_replay:
                sample_result = replay.sample(cfg.batch_size)
                s_b, a_b, r_b, s2_b, d_b, indices, weights_np = sample_result  # type: ignore[misc]
                weights: torch.Tensor | None = torch.tensor(weights_np, dtype=torch.float32, device=cfg.device)
            else:
                sample_result = replay.sample(cfg.batch_size)
                s_b, a_b, r_b, s2_b, d_b = sample_result  # type: ignore[misc]
                weights = None
                indices = None

            s_b = torch.tensor(s_b, dtype=torch.float32, device=cfg.device)
            a_b = torch.tensor(a_b, dtype=torch.int64, device=cfg.device)
            r_b = torch.tensor(r_b, dtype=torch.float32, device=cfg.device)
            s2_b = torch.tensor(s2_b, dtype=torch.float32, device=cfg.device)
            d_b = torch.tensor(d_b, dtype=torch.bool, device=cfg.device)

            with torch.no_grad():
                if cfg.use_double_dqn:
                    next_actions = q(s2_b).argmax(1)
                    target_q = tgt(s2_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    target_q = tgt(s2_b).max(1).values
                y = r_b + cfg.gamma * (~d_b).float() * target_q

            qvals = q(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

            if cfg.use_prioritized_replay and weights is not None and indices is not None:
                td_errors = (qvals - y).abs()
                loss = (weights * nn.functional.smooth_l1_loss(qvals, y, reduction='none')).mean()
                if hasattr(replay, 'update_priorities'):
                    replay.update_priorities(indices, td_errors.detach().cpu().numpy() + 1e-6)  # type: ignore[attr-defined]
            else:
                loss = nn.functional.smooth_l1_loss(qvals, y)

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
                epsilon=eps if not is_noisy else 0.0,
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
                    "config": cfg.__dict__,
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
            "config": cfg.__dict__,
        },
        os.path.join(ckpt_dir, "final.pt")
    )

    print(f"\nTraining complete! Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with YAML configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
