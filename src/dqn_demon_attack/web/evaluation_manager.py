"""
Evaluation manager for web-based model performance assessment.

Handles model evaluation via CLI subprocess with video recording
and comprehensive metrics collection for display.
"""

import os
import json
import time
import threading
import subprocess
from typing import Optional, Dict, List, Any
from pathlib import Path

import numpy as np


class EvaluationManager:
    """
    Manages model evaluation via CLI subprocess with file-based result tracking.

    Executes evaluation using CLI commands and collects results from
    JSON output and video files.
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
        Start evaluation via CLI subprocess.

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
            video_dir = os.path.normpath(video_dir)
            os.makedirs(video_dir, exist_ok=True)

            json_output = os.path.join(video_dir, "results.json")

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
                "video_dir": video_dir,
                "json_output": json_output
            }

            self.eval_thread = threading.Thread(
                target=self._eval_subprocess,
                args=(checkpoint_path, num_episodes, record_video, device, video_dir, json_output),
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

            completed_episodes = self.current_eval["completed_episodes"]

            if self.is_evaluating() and completed_episodes == 0:
                video_dir = self.current_eval["video_dir"]
                if os.path.exists(video_dir):
                    video_files = list(Path(video_dir).glob("*.mp4"))
                    completed_episodes = len(video_files)

            eval_data = {
                "eval_id": self.current_eval["eval_id"],
                "checkpoint_path": self.current_eval["checkpoint_path"],
                "num_episodes": self.current_eval["num_episodes"],
                "status": self.current_eval["status"],
                "start_time": self.current_eval["start_time"],
                "completed_episodes": completed_episodes,
                "results": {
                    "returns": list(self.current_eval["results"]["returns"]),
                    "lengths": list(self.current_eval["results"]["lengths"]),
                    "q_means": list(self.current_eval["results"]["q_means"]),
                    "q_stds": list(self.current_eval["results"]["q_stds"])
                },
                "videos": [dict(v) for v in self.current_eval["videos"]],
                "video_dir": self.current_eval["video_dir"]
            }

            if "end_time" in self.current_eval:
                eval_data["end_time"] = self.current_eval["end_time"]

            eval_data["is_active"] = self.is_evaluating()

            if completed_episodes > 0:
                elapsed = time.time() - eval_data["start_time"]
                eval_data["elapsed_time"] = elapsed
                eval_data["progress"] = completed_episodes / eval_data["num_episodes"]

                results = eval_data["results"]
                if results["returns"]:
                    eval_data["summary"] = {
                        "mean_return": float(np.mean(results["returns"])),
                        "std_return": float(np.std(results["returns"])),
                        "min_return": float(np.min(results["returns"])),
                        "max_return": float(np.max(results["returns"])),
                        "mean_length": float(np.mean(results["lengths"])),
                        "mean_q": float(np.mean([q for q in results["q_means"] if q is not None])) if [q for q in results["q_means"] if q is not None] else 0.0,
                    }

            return eval_data

    def _eval_subprocess(self, checkpoint_path: str, num_episodes: int,
                        record_video: bool, device: str, video_dir: str, json_output: str):
        """
        Run evaluation via CLI subprocess.

        Args:
            checkpoint_path: Path to model checkpoint.
            num_episodes: Number of episodes to evaluate.
            record_video: Whether to record videos.
            device: Device for inference.
            video_dir: Directory to save videos.
            json_output: Path to save JSON results.
        """
        try:
            cmd = [
                "uv", "run", "eval",
                "--ckpt", checkpoint_path,
                "--episodes", str(num_episodes),
                "--device", device,
                "--json-output", json_output
            ]

            if record_video:
                cmd.extend([
                    "--record",
                    "--output-dir", video_dir
                ])

            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Evaluation failed: {result.stderr}")

            if os.path.exists(json_output):
                with open(json_output, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                with self.lock:
                    if self.current_eval:
                        self.current_eval["results"]["returns"] = results["returns"]
                        self.current_eval["results"]["lengths"] = results["lengths"]
                        self.current_eval["results"]["q_means"] = results["q_means"]
                        self.current_eval["results"]["q_stds"] = results["q_stds"]
                        self.current_eval["completed_episodes"] = num_episodes

                        if record_video and results.get("videos"):
                            for ep_idx, video_path in enumerate(results["videos"]):
                                if ep_idx < len(results["returns"]):
                                    self.current_eval["videos"].append({
                                        "episode": ep_idx,
                                        "return": results["returns"][ep_idx],
                                        "length": results["lengths"][ep_idx],
                                        "path": video_path
                                    })

                        video_files = list(Path(video_dir).glob("*.mp4"))
                        for video_file in video_files:
                            if not any(v["path"] == str(video_file) for v in self.current_eval["videos"]):
                                self.current_eval["videos"].append({
                                    "episode": len(self.current_eval["videos"]),
                                    "return": 0.0,
                                    "length": 0,
                                    "path": str(video_file)
                                })

                        self.current_eval["status"] = "completed"
                        self.current_eval["end_time"] = time.time()
            else:
                raise RuntimeError("Evaluation JSON output not found")

        except Exception as e:
            with self.lock:
                if self.current_eval:
                    self.current_eval["status"] = "failed"
                    self.current_eval["error"] = str(e)
                    self.current_eval["end_time"] = time.time()
            raise
