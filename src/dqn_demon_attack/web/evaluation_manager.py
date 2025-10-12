"""
Evaluation manager for web-based model performance assessment.

Handles model loading, evaluation execution with video recording,
and comprehensive metrics collection for display.
"""

import os
import time
import threading
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from gymnasium.spaces import Discrete
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from dqn_demon_attack.agents import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN
from dqn_demon_attack.envs import make_env


class EvaluationManager:
    """
    Manages model evaluation with video recording and metrics tracking.

    Supports loading trained models, running evaluation episodes,
    recording videos, and collecting comprehensive performance metrics.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        self.current_eval: Optional[Dict[str, Any]] = None
        self.eval_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """
        List all available checkpoints for a training session.

        Args:
            session_id: Training session identifier.

        Returns:
            List of checkpoint info dictionaries.
        """
        run_dir = os.path.join(self.base_dir, session_id)
        ckpt_dir = os.path.join(run_dir, "checkpoints")

        if not os.path.exists(ckpt_dir):
            return []

        checkpoints = []
        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".pt"):
                path = os.path.join(ckpt_dir, fname)
                stat = os.stat(path)
                checkpoints.append({
                    "name": fname,
                    "path": path,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })

        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def start_evaluation(self, checkpoint_path: str, num_episodes: int = 10,
                        record_video: bool = True, device: str = "cuda") -> Dict[str, str]:
        """
        Start evaluation in background thread.

        Args:
            checkpoint_path: Path to model checkpoint.
            num_episodes: Number of episodes to evaluate.
            record_video: Whether to record evaluation videos.
            device: Device for inference.

        Returns:
            Dict containing eval_id and status.
        """
        with self.lock:
            if self.is_evaluating():
                raise RuntimeError("Evaluation already in progress")

            eval_id = f"eval_{int(time.time())}"
            video_dir = os.path.join(os.path.dirname(checkpoint_path), "..", "eval_videos", eval_id)
            os.makedirs(video_dir, exist_ok=True)

            self.current_eval = {
                "eval_id": eval_id,
                "checkpoint_path": checkpoint_path,
                "num_episodes": num_episodes,
                "status": "running",
                "start_time": time.time(),
                "completed_episodes": 0,
                "results": {
                    "returns": [],
                    "lengths": [],
                    "q_means": [],
                    "q_stds": []
                },
                "videos": [],
                "video_dir": video_dir
            }

            self.eval_thread = threading.Thread(
                target=self._eval_loop,
                args=(checkpoint_path, num_episodes, record_video, device, video_dir),
                daemon=True
            )
            self.eval_thread.start()

            return {"eval_id": eval_id, "status": "started"}

    def is_evaluating(self) -> bool:
        """Check if evaluation is currently active."""
        return self.eval_thread is not None and self.eval_thread.is_alive()

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current evaluation status and results.

        Returns:
            Dict with status and metrics, or None if not evaluating.
        """
        with self.lock:
            if self.current_eval is None:
                return None

            eval_data = self.current_eval.copy()
            eval_data["is_active"] = self.is_evaluating()

            if eval_data["completed_episodes"] > 0:
                elapsed = time.time() - eval_data["start_time"]
                eval_data["elapsed_time"] = elapsed
                eval_data["progress"] = eval_data["completed_episodes"] / eval_data["num_episodes"]

                results = eval_data["results"]
                if results["returns"]:
                    eval_data["summary"] = {
                        "mean_return": float(np.mean(results["returns"])),
                        "std_return": float(np.std(results["returns"])),
                        "min_return": float(np.min(results["returns"])),
                        "max_return": float(np.max(results["returns"])),
                        "mean_length": float(np.mean(results["lengths"])),
                        "mean_q": float(np.mean([q for q in results["q_means"] if q is not None])) if results["q_means"] else None,
                    }

            return eval_data

    def _load_model(self, ckpt_path: str, device: str):
        """
        Load trained model from checkpoint.

        Automatically detects model architecture from state dict.

        Args:
            ckpt_path: Path to checkpoint file.
            device: Device to load model on.

        Returns:
            Tuple of (model, n_actions).
        """
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"]

        has_adv = any("adv" in key for key in state_dict.keys())
        has_val = any("val" in key for key in state_dict.keys())
        has_noisy = any("sigma" in key for key in state_dict.keys())

        if has_adv and has_val:
            if has_noisy:
                model_class = NoisyDuelingDQN
            else:
                model_class = DuelingDQN
        else:
            if has_noisy:
                model_class = NoisyDQN
            else:
                model_class = DQN

        n_actions = None
        for key in sorted(state_dict.keys(), reverse=True):
            if "adv" in key and "3" in key and "bias" in key:
                n_actions = state_dict[key].shape[0]
                break
            elif "head" in key and "3" in key and "bias" in key:
                n_actions = state_dict[key].shape[0]
                break

        if n_actions is None:
            raise ValueError("Could not determine number of actions from checkpoint")

        q = model_class(n_actions).to(device)
        q.load_state_dict(state_dict)
        q.eval()

        return q, n_actions

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

    def _eval_loop(self, checkpoint_path: str, num_episodes: int,
                   record_video: bool, device: str, video_dir: str):
        """
        Main evaluation loop executed in background thread.

        Args:
            checkpoint_path: Path to model checkpoint.
            num_episodes: Number of episodes to evaluate.
            record_video: Whether to record videos.
            device: Device for inference.
            video_dir: Directory to save videos.
        """
        try:
            q, n_actions = self._load_model(checkpoint_path, device)

            for episode in range(num_episodes):
                if record_video:
                    env = make_env(render_mode="rgb_array")
                    env = RecordVideo(
                        env,
                        video_folder=video_dir,
                        episode_trigger=lambda e: e == 0,
                        name_prefix=f"episode_{episode}"
                    )
                else:
                    env = make_env(render_mode=None)

                env = RecordEpisodeStatistics(env)

                s, _ = env.reset()
                done = False
                ep_return = 0.0
                ep_len = 0
                q_vals_episode = []

                while not done:
                    s_t = self._to_tensor(s).to(device)
                    with torch.no_grad():
                        q_vals = q(s_t)
                        a = q_vals.argmax(1).item()
                        q_vals_episode.append({
                            "mean": float(q_vals.mean().item()),
                            "std": float(q_vals.std().item())
                        })

                    s, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    ep_return += float(info.get("raw_reward", r))
                    ep_len += 1

                env.close()

                q_mean = np.mean([qv["mean"] for qv in q_vals_episode]) if q_vals_episode else None
                q_std = np.mean([qv["std"] for qv in q_vals_episode]) if q_vals_episode else None

                with self.lock:
                    if self.current_eval:
                        self.current_eval["completed_episodes"] = episode + 1
                        self.current_eval["results"]["returns"].append(ep_return)
                        self.current_eval["results"]["lengths"].append(ep_len)
                        self.current_eval["results"]["q_means"].append(q_mean)
                        self.current_eval["results"]["q_stds"].append(q_std)

                        if record_video:
                            video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
                            if video_files:
                                latest_video = sorted(video_files)[-1]
                                self.current_eval["videos"].append({
                                    "episode": episode,
                                    "return": ep_return,
                                    "length": ep_len,
                                    "path": os.path.join(video_dir, latest_video)
                                })

            with self.lock:
                if self.current_eval:
                    self.current_eval["status"] = "completed"
                    self.current_eval["end_time"] = time.time()

        except Exception as e:
            with self.lock:
                if self.current_eval:
                    self.current_eval["status"] = "failed"
                    self.current_eval["error"] = str(e)
            raise
