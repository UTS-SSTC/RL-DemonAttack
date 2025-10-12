"""
Training manager for web-based DQN training interface.

Handles training execution, progress tracking, breakthrough video recording,
and metrics collection for visualization.
"""

import os
import time
import random
import threading
from typing import Optional, Dict, List, Any
from collections import deque

import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.wrappers import RecordVideo
from torch import nn

from dqn_demon_attack.agents import DQN, Replay
from dqn_demon_attack.envs import make_env, RewardConfig
from dqn_demon_attack.utils import CSVLogger


class TrainingConfig:
    """Configuration for training session."""

    def __init__(self, **kwargs):
        self.exp_name = kwargs.get("exp_name", "web_train")
        self.total_steps = kwargs.get("total_steps", 10000)
        self.eval_every = kwargs.get("eval_every", 2000)
        self.seed = kwargs.get("seed", 0)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.terminal_on_life_loss = kwargs.get("terminal_on_life_loss", True)
        self.screen_size = kwargs.get("screen_size", 84)
        self.frame_stack = kwargs.get("frame_stack", 4)

        self.reward_mode = kwargs.get("reward_mode", "clip")
        self.gamma = kwargs.get("gamma", 0.99)
        self.lr = kwargs.get("lr", 1e-4)
        self.batch_size = kwargs.get("batch_size", 32)
        self.replay_size = kwargs.get("replay_size", 50000)
        self.warmup = kwargs.get("warmup", 1000)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.grad_clip = kwargs.get("grad_clip", 10.0)

        self.eps_start = kwargs.get("eps_start", 1.0)
        self.eps_end = kwargs.get("eps_end", 0.1)
        self.eps_decay_steps = kwargs.get("eps_decay_steps", 100000)

        self.max_videos = kwargs.get("max_videos", 10)
        self.breakthrough_threshold = kwargs.get("breakthrough_threshold", 0.15)


class TrainingManager:
    """
    Manages DQN training sessions with real-time monitoring and video recording.

    Supports concurrent training execution, breakthrough detection, automatic
    video recording of high-performing episodes, and metrics tracking.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        self.current_session: Optional[Dict[str, Any]] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def start_training(self, config: TrainingConfig) -> Dict[str, str]:
        """
        Start a new training session in background thread.

        Args:
            config: Training configuration parameters.

        Returns:
            Dict containing session_id and run_dir.
        """
        with self.lock:
            if self.is_training():
                raise RuntimeError("Training already in progress")

            session_id = f"{config.exp_name}_{int(time.time())}"
            run_dir = os.path.join(self.base_dir, session_id)
            os.makedirs(run_dir, exist_ok=True)

            self.current_session = {
                "session_id": session_id,
                "run_dir": run_dir,
                "config": config,
                "status": "running",
                "start_time": time.time(),
                "current_step": 0,
                "total_steps": config.total_steps,
                "logs": deque(maxlen=1000),
                "metrics": {
                    "episodes": [],
                    "returns": [],
                    "losses": [],
                    "q_values": [],
                    "steps": [],
                },
                "videos": [],
                "best_return": float("-inf"),
                "checkpoints": [],
            }

            self.stop_flag.clear()
            self.training_thread = threading.Thread(
                target=self._train_loop,
                args=(config, run_dir),
                daemon=True
            )
            self.training_thread.start()

            return {"session_id": session_id, "run_dir": run_dir}

    def stop_training(self):
        """Request training to stop gracefully."""
        self.stop_flag.set()

    def is_training(self) -> bool:
        """Check if training is currently active."""
        return self.training_thread is not None and self.training_thread.is_alive()

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current training status and metrics.

        Returns:
            Dict with status, progress, logs, and metrics, or None if not training.
        """
        with self.lock:
            if self.current_session is None:
                return None

            session = self.current_session.copy()
            session["logs"] = list(session["logs"])
            session["is_active"] = self.is_training()

            if session["current_step"] > 0:
                elapsed = time.time() - session["start_time"]
                session["elapsed_time"] = elapsed
                session["progress"] = session["current_step"] / session["total_steps"]

            return session

    def get_training_curves(self, session_id: str) -> Dict[str, Any]:
        """
        Load training curves from completed session.

        Args:
            session_id: ID of training session.

        Returns:
            Dict containing metrics data for plotting.
        """
        run_dir = os.path.join(self.base_dir, session_id)
        csv_path = os.path.join(run_dir, "train_log.csv")

        if not os.path.exists(csv_path):
            return {"error": "Training log not found"}

        import csv
        data = {
            "steps": [],
            "episodes": [],
            "returns_raw": [],
            "returns_shaped": [],
            "lengths": [],
            "losses": [],
            "q_values": [],
            "epsilon": [],
        }

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["steps"].append(int(row["step"]))
                data["episodes"].append(int(row["episode"]))
                data["returns_raw"].append(float(row["ep_return_raw"]))
                data["returns_shaped"].append(float(row["ep_return_shaped"]))
                data["lengths"].append(int(row["ep_len"]))

                loss = row["loss"]
                data["losses"].append(float(loss) if loss != "nan" else None)

                q_val = row["q_mean"]
                data["q_values"].append(float(q_val) if q_val != "nan" else None)

                data["epsilon"].append(float(row["epsilon"]))

        return data

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _to_tensor(self, obs):
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

        arr = arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr[None, ...])

    def _should_record_video(self, current_return: float, best_return: float, threshold: float) -> bool:
        """Determine if current episode represents a breakthrough."""
        if best_return == float("-inf"):
            return False
        improvement = (current_return - best_return) / (abs(best_return) + 1e-6)
        return improvement >= threshold

    def _log_message(self, message: str):
        """Add message to training logs."""
        with self.lock:
            if self.current_session:
                self.current_session["logs"].append({
                    "time": time.time(),
                    "message": message
                })

    def _record_breakthrough_video(self, q, env_maker, device: str, run_dir: str,
                                   step: int, reward: float) -> str:
        """
        Record video of breakthrough episode.

        Args:
            q: Q-network model.
            env_maker: Function to create environment.
            device: Device for inference.
            run_dir: Directory to save video.
            step: Current training step.
            reward: Episode reward achieved.

        Returns:
            Path to recorded video file.
        """
        video_dir = os.path.join(run_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        env = env_maker()
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda e: e == 0,
            name_prefix=f"breakthrough_step{step}_reward{int(reward)}"
        )

        q.eval()
        s, _ = env.reset()
        done = False

        while not done:
            s_t = self._to_tensor(s).to(device)
            with torch.no_grad():
                a = q(s_t).argmax(1).item()
            s, r, term, trunc, info = env.step(a)
            done = term or trunc

        env.close()
        q.train()

        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        if video_files:
            return os.path.join(video_dir, video_files[-1])
        return ""

    def _train_loop(self, cfg: TrainingConfig, run_dir: str):
        """
        Main training loop executed in background thread.

        Args:
            cfg: Training configuration.
            run_dir: Directory for saving outputs.
        """
        try:
            self._set_seed(cfg.seed)

            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            log = CSVLogger(
                os.path.join(run_dir, "train_log.csv"),
                fieldnames=[
                    "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
                    "loss", "q_mean", "epsilon", "replay_size"
                ]
            )

            reward_cfg = RewardConfig(mode=cfg.reward_mode)
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
            s_t = self._to_tensor(s).to(cfg.device)
            ep_return_raw, ep_return_shaped, ep_len, episode = 0.0, 0.0, 0, 0
            loss_val, q_mean = float("nan"), float("nan")

            recent_returns = deque(maxlen=10)

            self._log_message(f"Training started: {cfg.total_steps} steps")

            for step in range(1, cfg.total_steps + 1):
                if self.stop_flag.is_set():
                    self._log_message("Training stopped by user")
                    break

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

                replay.push(s_t.cpu().numpy()[0], a, r, self._to_tensor(s2).cpu().numpy()[0], done)
                s_t = self._to_tensor(s2).to(cfg.device)

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
                        epsilon=eps,
                        replay_size=len(replay)
                    )

                    with self.lock:
                        if self.current_session:
                            self.current_session["current_step"] = step
                            self.current_session["metrics"]["episodes"].append(episode)
                            self.current_session["metrics"]["returns"].append(ep_return_raw)
                            self.current_session["metrics"]["steps"].append(step)
                            if loss_val == loss_val:
                                self.current_session["metrics"]["losses"].append(loss_val)
                            if q_mean == q_mean:
                                self.current_session["metrics"]["q_values"].append(q_mean)

                    recent_returns.append(ep_return_raw)
                    avg_recent = np.mean(recent_returns) if recent_returns else 0.0

                    with self.lock:
                        if self.current_session and ep_return_raw > self.current_session["best_return"]:
                            improvement = ep_return_raw - self.current_session["best_return"]

                            if self._should_record_video(ep_return_raw, self.current_session["best_return"], cfg.breakthrough_threshold):
                                self._log_message(f"Breakthrough detected at step {step}: reward {ep_return_raw:.1f} (improvement: {improvement:.1f})")

                                if len(self.current_session["videos"]) >= cfg.max_videos:
                                    oldest_video = self.current_session["videos"].pop(0)
                                    if os.path.exists(oldest_video["path"]):
                                        os.remove(oldest_video["path"])

                                env_maker = lambda: make_env(
                                    render_mode="rgb_array",
                                    stack=cfg.frame_stack,
                                    screen_size=cfg.screen_size,
                                    terminal_on_life_loss=cfg.terminal_on_life_loss,
                                    reward_cfg=reward_cfg
                                )

                                video_path = self._record_breakthrough_video(q, env_maker, cfg.device, run_dir, step, ep_return_raw)

                                if video_path:
                                    self.current_session["videos"].append({
                                        "step": step,
                                        "reward": ep_return_raw,
                                        "path": video_path,
                                        "timestamp": time.time()
                                    })

                            self.current_session["best_return"] = ep_return_raw

                    self._log_message(f"Episode {episode} | Step {step} | Return: {ep_return_raw:.1f} | Avg: {avg_recent:.1f}")

                    ep_return_raw, ep_return_shaped, ep_len = 0.0, 0.0, 0
                    s, _ = env.reset()
                    s_t = self._to_tensor(s).to(cfg.device)

                if step % cfg.eval_every == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
                    torch.save(
                        {
                            "step": step,
                            "model": q.state_dict(),
                            "optimizer": opt.state_dict(),
                            "cfg": vars(cfg),
                        },
                        ckpt_path
                    )

                    with self.lock:
                        if self.current_session:
                            self.current_session["checkpoints"].append({
                                "step": step,
                                "path": ckpt_path
                            })

                    self._log_message(f"Checkpoint saved at step {step}")

            final_path = os.path.join(ckpt_dir, "final.pt")
            torch.save(
                {
                    "step": cfg.total_steps,
                    "model": q.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": vars(cfg),
                },
                final_path
            )

            with self.lock:
                if self.current_session:
                    self.current_session["checkpoints"].append({
                        "step": cfg.total_steps,
                        "path": final_path
                    })

            log.close()
            env.close()

            with self.lock:
                if self.current_session:
                    self.current_session["status"] = "completed"

            self._log_message("Training completed successfully")

        except Exception as e:
            self._log_message(f"Training failed: {str(e)}")
            with self.lock:
                if self.current_session:
                    self.current_session["status"] = "failed"
            raise
