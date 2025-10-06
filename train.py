import os
import random

import numpy as np
import torch
from torch import nn

from envs import make_env, RewardConfig
from logger import CSVLogger
from models import DQN
from replay import Replay


class Cfg:
    # Run
    exp_name = "exp1"
    total_steps = 500_000
    eval_every = 50_000
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "mps"

    # Env
    terminal_on_life_loss = True
    screen_size = 84
    frame_stack = 4

    # Reward
    reward_cfg = RewardConfig(mode="clip")  # "clip"|"scaled"|"raw"
    # reward_cfg = RewardConfig(mode="scaled", scale_divisor=100.0)

    # DQN
    gamma = 0.99
    lr = 1e-4
    batch_size = 32
    replay_size = 50_000
    warmup = 5_000
    target_update_freq = 10_000
    grad_clip = 10.0

    # epsilon schedule
    eps_start = 1.0
    eps_end = 0.1
    eps_decay_steps = 1_000_000  # linear to 0.1


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def main(cfg: Cfg = Cfg()):
    set_seed(cfg.seed)
    run_dir = os.path.join("runs", cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log = CSVLogger(os.path.join(run_dir, "train_log.csv"),
                    fieldnames=["step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
                                "loss", "q_mean", "epsilon", "replay_size"])

    # Env
    env = make_env(render_mode=None,
                   stack=cfg.frame_stack,
                   screen_size=cfg.screen_size,
                   terminal_on_life_loss=cfg.terminal_on_life_loss,
                   reward_cfg=cfg.reward_cfg)
    n_actions = env.action_space.n

    # Model
    q = DQN(n_actions).to(cfg.device)
    tgt = DQN(n_actions).to(cfg.device);
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_size)

    # State
    s, _ = env.reset(seed=cfg.seed)
    s_t = to_tensor(s).to(cfg.device)
    ep_return_raw, ep_return_shaped, ep_len, episode = 0.0, 0.0, 0, 0
    loss_val, q_mean = float("nan"), float("nan")

    for step in range(1, cfg.total_steps + 1):
        # epsilon
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * (step / cfg.eps_decay_steps))

        # act
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

        # replay.push(np.array(s_t.cpu())[0], a, r, np.array(to_tensor(s2))[0], done)
        replay.push(s_t.cpu().numpy()[0], a, r, to_tensor(s2).cpu().numpy()[0], done)
        s_t = to_tensor(s2).to(cfg.device)

        # Learn
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

        # Episode end
        if done:
            episode += 1
            log.log(step=step, episode=episode,
                    ep_return_raw=ep_return_raw,
                    ep_return_shaped=ep_return_shaped,
                    ep_len=ep_len,
                    loss=loss_val, q_mean=q_mean, epsilon=eps,
                    replay_size=len(replay))
            # reset counters
            ep_return_raw, ep_return_shaped, ep_len = 0.0, 0.0, 0
            s, _ = env.reset()
            s_t = to_tensor(s).to(cfg.device)

        # Periodic eval (prints to stdout)
        if step % cfg.eval_every == 0:
            avg, std, avg_len = evaluate(q, device=cfg.device, episodes=10)
            # Also save checkpoint
            torch.save({
                "step": step,
                "model": q.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": vars(cfg),
                "eval_mean": avg,
                "eval_std": std,
            }, os.path.join(ckpt_dir, f"step_{step}.pt"))
            print(f"[Eval @ {step}] raw return {avg:.1f}±{std:.1f} | len {avg_len:.0f}")

    log.close()
    # final save
    torch.save({
        "step": cfg.total_steps,
        "model": q.state_dict(),
        "optimizer": opt.state_dict(),
        "cfg": vars(cfg),
    }, os.path.join(ckpt_dir, f"final.pt"))


def evaluate(q, device="cuda", episodes=10):
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
        rets.append(ep_ret);
        lens.append(ep_len)
    q.train()
    import numpy as np
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(lens))


if __name__ == "__main__":
    main()
