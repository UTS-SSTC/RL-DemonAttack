"""
Training manager for web-based DQN training interface.

Handles training execution via CLI subprocess, progress tracking through
file monitoring, and metrics collection for visualization.
"""

import os
import csv
import time
import subprocess
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass

from dqn_demon_attack.utils import TrainingConfig, save_config


@dataclass
class WebTrainingConfig(TrainingConfig):
    """Extended training configuration for web interface."""

    max_videos: int = 10
    breakthrough_threshold: float = 0.15

    def __init__(self, **kwargs):
        valid_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}

        base_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        super().__init__(**base_kwargs)

        self.max_videos = kwargs.get("max_videos", 10)
        self.breakthrough_threshold = kwargs.get("breakthrough_threshold", 0.15)


class TrainingManager:
    """
    Manages DQN training sessions via CLI subprocess with file-based monitoring.

    Executes training using CLI commands and monitors progress by reading
    CSV logs, text logs, and scanning for videos/checkpoints.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        self.current_session: Optional[Dict[str, Any]] = None
        self.process: Optional[subprocess.Popen] = None

    def start_training(self, config: WebTrainingConfig) -> Dict[str, str]:
        """
        Start a new training session via CLI subprocess.

        Args:
            config: Training configuration parameters.

        Returns:
            Dict containing session_id and run_dir.
        """
        if self.is_training():
            raise RuntimeError("Training already in progress")

        session_id = f"{config.exp_name}_{int(time.time())}"
        run_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(run_dir, exist_ok=True)

        config.exp_name = session_id
        config_path = os.path.join(run_dir, "config.yaml")
        save_config(config, config_path)

        log_file = os.path.join(run_dir, "train.log")
        config.log_file = log_file

        cmd = [
            "uv", "run", "train",
            "--config", config_path,
        ]

        if config.record_breakthrough:
            cmd.extend([
                "--record-breakthrough",
                "--breakthrough-threshold", str(config.breakthrough_threshold),
                "--max-videos", str(config.max_videos),
            ])

        cmd.extend([
            "--log-file", log_file
        ])

        log_file_handle = open(log_file, 'w', encoding='utf-8')

        self.process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=log_file_handle,
            stderr=subprocess.STDOUT,
            text=True
        )

        self.current_session = {
            "session_id": session_id,
            "run_dir": run_dir,
            "config": config,
            "status": "running",
            "start_time": time.time(),
            "total_steps": config.total_steps,
            "process": self.process,
            "log_file_handle": log_file_handle,
        }

        return {"session_id": session_id, "run_dir": run_dir}

    def stop_training(self):
        """Terminate training process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.current_session and "log_file_handle" in self.current_session:
            try:
                self.current_session["log_file_handle"].close()
            except Exception:
                pass

    def is_training(self) -> bool:
        """Check if training process is currently active."""
        return self.process is not None and self.process.poll() is None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current training status by reading files.

        Returns:
            Dict with status, progress, logs, metrics, and videos.
        """
        if self.current_session is None:
            return None

        session = self.current_session.copy()
        run_dir = session["run_dir"]

        if "process" in session:
            del session["process"]
        if "config" in session:
            del session["config"]
        if "log_file_handle" in session:
            del session["log_file_handle"]

        session["is_active"] = self.is_training()

        if not self.is_training() and session["status"] == "running":
            if self.current_session and "log_file_handle" in self.current_session:
                try:
                    self.current_session["log_file_handle"].close()
                except Exception:
                    pass

            if self.process and self.process.returncode == 0:
                session["status"] = "completed"
            else:
                session["status"] = "failed"

        csv_path = os.path.join(run_dir, "train_log.csv")
        current_step = 0
        metrics = {
            "episodes": [],
            "returns": [],
            "losses": [],
            "q_values": [],
            "steps": [],
        }

        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        current_step = int(rows[-1]["step"])
                        for row in rows[-100:]:
                            metrics["episodes"].append(int(row["episode"]))
                            metrics["returns"].append(float(row["ep_return_raw"]))
                            metrics["steps"].append(int(row["step"]))
                            loss = row.get("loss", "nan")
                            metrics["losses"].append(float(loss) if loss != "nan" else None)
                            q_val = row.get("q_mean", "nan")
                            metrics["q_values"].append(float(q_val) if q_val != "nan" else None)
            except Exception as e:
                print(f"Warning: Failed to read CSV: {e}")

        session["current_step"] = current_step
        session["metrics"] = metrics

        if current_step > 0:
            elapsed = time.time() - session["start_time"]
            session["elapsed_time"] = elapsed
            session["progress"] = current_step / session["total_steps"]
        else:
            session["progress"] = 0.0

        log_path = os.path.join(run_dir, "train.log")
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logs = [{"message": line.strip()} for line in lines[-50:] if line.strip()]
            except Exception as e:
                print(f"Warning: Failed to read log file: {e}")

        session["logs"] = logs

        video_dir = os.path.join(run_dir, "videos")
        videos = []
        if os.path.exists(video_dir):
            try:
                for video_file in Path(video_dir).glob("*.mp4"):
                    videos.append({
                        "path": str(video_file),
                        "name": video_file.name,
                        "size": video_file.stat().st_size,
                    })
            except Exception as e:
                print(f"Warning: Failed to scan videos: {e}")

        session["videos"] = videos

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        checkpoints = []
        if os.path.exists(ckpt_dir):
            try:
                for ckpt_file in Path(ckpt_dir).glob("*.pt"):
                    checkpoints.append({
                        "path": str(ckpt_file),
                        "name": ckpt_file.name,
                    })
            except Exception as e:
                print(f"Warning: Failed to scan checkpoints: {e}")

        session["checkpoints"] = checkpoints

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

        with open(csv_path, "r", encoding='utf-8') as f:
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
