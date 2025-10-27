"""
Training manager for web-based DQN training interface.

Handles training execution, progress tracking, breakthrough video recording,
and metrics collection for visualization.
"""

import os
import time
import threading
from typing import Optional, Dict, List, Any
from collections import deque
from dataclasses import dataclass, asdict

import torch
from gymnasium.wrappers import RecordVideo

from dqn_demon_attack.envs import make_env, RewardConfig
from dqn_demon_attack.utils import TrainingConfig, CSVLogger
from dqn_demon_attack.utils.training_utils import train_dqn, to_tensor


@dataclass
class WebTrainingConfig(TrainingConfig):
    """Extended training configuration for web interface."""

    max_videos: int = 10
    breakthrough_threshold: float = 0.15

    def __init__(self, **kwargs):
        valid_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        web_fields = {"max_videos", "breakthrough_threshold"}

        base_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        super().__init__(**base_kwargs)

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

    def start_training(self, config: WebTrainingConfig) -> Dict[str, str]:
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
            s_t = to_tensor(s).to(device)
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

    def _train_loop(self, cfg: WebTrainingConfig, run_dir: str):
        """
        Main training loop executed in background thread.

        Args:
            cfg: Training configuration.
            run_dir: Directory for saving outputs.
        """
        try:
            self._log_message(f"Training started: {cfg.total_steps} steps")

            log = CSVLogger(
                os.path.join(run_dir, "train_log.csv"),
                fieldnames=[
                    "step", "episode", "ep_return_raw", "ep_return_shaped", "ep_len",
                    "loss", "q_mean", "epsilon", "replay_size", "lr"
                ]
            )

            recent_returns = deque(maxlen=10)
            current_q_network = None

            def on_step_callback(step: int, metrics: Dict[str, Any]) -> bool:
                with self.lock:
                    if self.current_session:
                        self.current_session["current_step"] = step

                return self.stop_flag.is_set()

            def on_episode_callback(episode: int, step: int, ep_return_raw: float,
                                   ep_return_shaped: float, ep_len: int):
                nonlocal current_q_network

                with self.lock:
                    if self.current_session:
                        self.current_session["metrics"]["episodes"].append(episode)
                        self.current_session["metrics"]["returns"].append(ep_return_raw)
                        self.current_session["metrics"]["steps"].append(step)

                recent_returns.append(ep_return_raw)
                avg_recent = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0

                with self.lock:
                    if self.current_session and ep_return_raw > self.current_session["best_return"]:
                        improvement = ep_return_raw - self.current_session["best_return"]

                        if self._should_record_video(ep_return_raw, self.current_session["best_return"],
                                                     cfg.breakthrough_threshold):
                            self._log_message(f"Breakthrough detected at step {step}: reward {ep_return_raw:.1f} (improvement: {improvement:.1f})")

                            if len(self.current_session["videos"]) >= cfg.max_videos:
                                oldest_video = self.current_session["videos"].pop(0)
                                if os.path.exists(oldest_video["path"]):
                                    os.remove(oldest_video["path"])

                            reward_cfg = RewardConfig(mode=cfg.reward_mode)
                            env_maker = lambda: make_env(
                                render_mode="rgb_array",
                                stack=cfg.frame_stack,
                                screen_size=cfg.screen_size,
                                terminal_on_life_loss=cfg.terminal_on_life_loss,
                                reward_cfg=reward_cfg
                            )

                            if current_q_network is not None:
                                video_path = self._record_breakthrough_video(
                                    current_q_network, env_maker, cfg.device, run_dir, step, ep_return_raw
                                )

                                if video_path:
                                    self.current_session["videos"].append({
                                        "step": step,
                                        "reward": ep_return_raw,
                                        "path": video_path,
                                        "timestamp": time.time()
                                    })

                        self.current_session["best_return"] = ep_return_raw

                self._log_message(f"Episode {episode} | Step {step} | Return: {ep_return_raw:.1f} | Avg: {avg_recent:.1f}")

            def on_checkpoint_callback(step: int, ckpt_path: str):
                with self.lock:
                    if self.current_session:
                        self.current_session["checkpoints"].append({
                            "step": step,
                            "path": ckpt_path
                        })
                self._log_message(f"Checkpoint saved at step {step}")

            result = train_dqn(
                cfg=cfg,
                log=log,
                run_dir=run_dir,
                on_step_callback=on_step_callback,
                on_episode_callback=on_episode_callback,
                on_checkpoint_callback=on_checkpoint_callback
            )

            current_q_network = result["final_model"]

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
