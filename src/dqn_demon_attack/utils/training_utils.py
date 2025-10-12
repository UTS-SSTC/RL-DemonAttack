"""
Shared training utilities for DQN agents.

Provides common functions for tensor conversion, model creation, seeding,
and core training loop logic used across training scripts and web interface.
"""

import os
import random
from typing import Tuple, Callable, Optional, Dict, Any

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch import nn

from dqn_demon_attack.agents import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN, Replay, PrioritizedReplay
from dqn_demon_attack.envs import make_env, RewardConfig
from dqn_demon_attack.utils.config import TrainingConfig
from dqn_demon_attack.utils.logger import CSVLogger


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def to_tensor(obs) -> torch.Tensor:
    """
    Convert observation to normalized PyTorch tensor.

    Handles various observation shapes and converts to [1, C, H, W] format
    with values normalized to [0, 1].

    Args:
        obs: Numpy array observation from environment.

    Returns:
        PyTorch tensor of shape [1, C, H, W] with float32 values in [0, 1].

    Raises:
        RuntimeError: If observation shape is not recognized.
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
            raise RuntimeError(f"Unrecognized obs shape {arr.shape}")
    else:
        raise RuntimeError(f"Unrecognized obs ndim {arr.ndim}")

    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr[None, ...])


def create_model(model_type: str, n_actions: int) -> nn.Module:
    """
    Create DQN model based on architecture type.

    Args:
        model_type: Model architecture name (DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN).
        n_actions: Number of actions in the environment.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    model_map = {
        "DQN": DQN,
        "DuelingDQN": DuelingDQN,
        "NoisyDQN": NoisyDQN,
        "NoisyDuelingDQN": NoisyDuelingDQN,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_map[model_type](n_actions)


def load_model(ckpt_path: str, device: str = "cpu") -> Tuple[nn.Module, int]:
    """
    Load trained model from checkpoint.

    Automatically detects model architecture from state dict keys.

    Args:
        ckpt_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (loaded_model, n_actions).

    Raises:
        ValueError: If number of actions cannot be determined from checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]

    has_adv = any("adv" in key for key in state_dict.keys())
    has_val = any("val" in key for key in state_dict.keys())
    has_noisy = any("sigma" in key for key in state_dict.keys())

    if has_adv and has_val:
        model_class = NoisyDuelingDQN if has_noisy else DuelingDQN
    else:
        model_class = NoisyDQN if has_noisy else DQN

    n_actions = None
    for key in sorted(state_dict.keys(), reverse=True):
        if ("adv" in key or "head" in key) and "3" in key and "bias" in key:
            n_actions = state_dict[key].shape[0]
            break

    if n_actions is None:
        raise ValueError("Could not determine number of actions from checkpoint")

    model = model_class(n_actions).to(device)
    model.load_state_dict(state_dict)

    return model, n_actions


def evaluate_model(model: nn.Module, device: str = "cuda", episodes: int = 10,
                   stack: int = 4, screen_size: int = 84) -> Tuple[float, float, float]:
    """
    Evaluate model performance over multiple episodes.

    Args:
        model: Q-network to evaluate.
        device: Device for inference.
        episodes: Number of episodes to evaluate.
        stack: Frame stack count for environment.
        screen_size: Screen size for environment.

    Returns:
        Tuple of (mean_return, std_return, mean_length).
    """
    env = make_env(render_mode=None, stack=stack, screen_size=screen_size)
    model.eval()
    returns, lengths = [], []

    for _ in range(episodes):
        s, _ = env.reset()
        s_t = to_tensor(s).to(device)
        done, ep_return, ep_len = False, 0.0, 0

        while not done:
            with torch.no_grad():
                a = model(s_t).argmax(1).item()
            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            s_t = to_tensor(s2).to(device)
            ep_return += float(info.get("raw_reward", r))
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()
    model.train()

    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def train_dqn(
    cfg: TrainingConfig,
    log: CSVLogger,
    run_dir: str,
    on_step_callback: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
    on_episode_callback: Optional[Callable[[int, int, float, float, int], None]] = None,
    on_checkpoint_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Core DQN training loop with callback support.

    Args:
        cfg: Training configuration.
        log: CSV logger instance.
        run_dir: Directory for saving checkpoints.
        on_step_callback: Optional callback after each step. Returns True to stop training.
        on_episode_callback: Optional callback after each episode.
        on_checkpoint_callback: Optional callback after checkpoint save.

    Returns:
        Dict containing training statistics and final model state.
    """
    set_seed(cfg.seed)

    reward_cfg_kwargs: Dict[str, Any] = {"mode": cfg.reward_mode}
    if cfg.use_reward_shaping:
        if cfg.use_life_penalty:
            reward_cfg_kwargs['use_life_penalty'] = True
            reward_cfg_kwargs['life_penalty'] = cfg.life_penalty
        if cfg.use_streak_bonus:
            reward_cfg_kwargs['use_streak_bonus'] = True
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
    best_eval_return = float("-inf")

    is_noisy = "Noisy" in cfg.model_type

    for step in range(1, cfg.total_steps + 1):
        if is_noisy and hasattr(q, 'reset_noise'):
            q.reset_noise()  # type: ignore[attr-defined]
            tgt.reset_noise()  # type: ignore[attr-defined]

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
                s_b, a_b, r_b, s2_b, d_b, indices, weights_np = sample_result  # type: ignore
                weights: Optional[torch.Tensor] = torch.tensor(weights_np, dtype=torch.float32, device=cfg.device)
            else:
                sample_result = replay.sample(cfg.batch_size)
                s_b, a_b, r_b, s2_b, d_b = sample_result  # type: ignore
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

        if on_step_callback:
            should_stop = on_step_callback(step, {
                "episode": episode,
                "ep_return_raw": ep_return_raw,
                "ep_return_shaped": ep_return_shaped,
                "ep_len": ep_len,
                "loss": loss_val,
                "q_mean": q_mean,
                "epsilon": eps if not is_noisy else 0.0,
                "replay_size": len(replay)
            })
            if should_stop:
                break

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

            if on_episode_callback:
                on_episode_callback(episode, step, ep_return_raw, ep_return_shaped, ep_len)

            ep_return_raw, ep_return_shaped, ep_len = 0.0, 0.0, 0
            s, _ = env.reset()
            s_t = to_tensor(s).to(cfg.device)

        if step % cfg.eval_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            avg, std, avg_len = evaluate_model(q, device=cfg.device, episodes=10)
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": q.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg.__dict__,
                    "eval_mean": avg,
                    "eval_std": std,
                },
                ckpt_path
            )

            if avg > best_eval_return:
                best_eval_return = avg
                torch.save(
                    {
                        "step": step,
                        "model": q.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": cfg.__dict__,
                        "eval_mean": avg,
                        "eval_std": std,
                    },
                    os.path.join(ckpt_dir, "best.pt")
                )

            if on_checkpoint_callback:
                on_checkpoint_callback(step, ckpt_path)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "step": cfg.total_steps,
            "model": q.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg.__dict__,
        },
        final_path
    )

    log.close()
    env.close()

    return {
        "final_model": q,
        "target_model": tgt,
        "optimizer": opt,
        "best_eval_return": best_eval_return
    }