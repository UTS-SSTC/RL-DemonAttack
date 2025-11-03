"""
Flask web application for DQN training and evaluation GUI.

Provides modern REST API endpoints for training management, model evaluation,
video playback, and real-time metrics visualization with an enhanced UI.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from dqn_demon_attack.web.training_manager import TrainingManager, WebTrainingConfig
from dqn_demon_attack.web.evaluation_manager import EvaluationManager


app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))

CORS(app, resources={r"/api/*": {"origins": "*"}})

training_manager = TrainingManager(base_dir="runs")
evaluation_manager = EvaluationManager(base_dir="runs")


@app.route("/")
def index():
    """
    Render main GUI page.

    Returns:
        Rendered HTML template for the dashboard.
    """
    return render_template("index.html")


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """
    Start a new training session.

    Expected JSON body:
        exp_name: Experiment name.
        total_steps: Total training steps.
        eval_every: Evaluation interval.
        model_type: Model architecture type.
        device: Computing device (cuda/cpu).
        record_breakthrough: Enable breakthrough video recording (default: True).
        breakthrough_threshold: Performance improvement threshold (default: 0.15).
        max_videos: Maximum number of breakthrough videos (default: 10).
        Other training config parameters.

    Returns:
        JSON response with session_id and run_dir on success.
    """
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        if "record_breakthrough" not in data:
            data["record_breakthrough"] = True

        config = WebTrainingConfig(**data)
        result = training_manager.start_training(config)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """
    Stop current training session.

    Returns:
        JSON response indicating success or failure.
    """
    try:
        training_manager.stop_training()
        return jsonify({"success": True, "message": "Training stop requested"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """
    Get current training status and metrics.

    Returns:
        JSON with training progress, metrics, logs, and videos.
    """
    status = training_manager.get_status()
    if status is None:
        return jsonify({"is_training": False})

    safe_status = dict(status)
    if "config" in safe_status:
        del safe_status["config"]

    return jsonify({"is_training": safe_status.get("is_active", False), **safe_status})


@app.route("/api/training/curves/<session_id>", methods=["GET"])
def training_curves(session_id):
    """
    Get training curves for visualization.

    Args:
        session_id: Training session identifier.

    Returns:
        JSON with arrays of metrics (steps, returns, losses, Q-values, epsilon).
    """
    try:
        curves = training_manager.get_training_curves(session_id)
        return jsonify({"success": True, "data": curves})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """
    List all training sessions.

    Returns:
        JSON array of session info (session_id, path, created timestamp).
    """
    try:
        sessions = []
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            for session_name in os.listdir(runs_dir):
                session_path = os.path.join(runs_dir, session_name)
                if os.path.isdir(session_path):
                    stat = os.stat(session_path)
                    sessions.append({
                        "session_id": session_name,
                        "path": session_path,
                        "created": stat.st_ctime
                    })

        sessions.sort(key=lambda x: x["created"], reverse=True)
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/sessions/<session_id>/config", methods=["GET"])
def get_session_config(session_id):
    """
    Get configuration for a specific session.

    Args:
        session_id: Training session identifier.

    Returns:
        JSON with session configuration.
    """
    try:
        config_path = os.path.join("runs", session_id, "config.yaml")
        if not os.path.exists(config_path):
            return jsonify({"success": False, "error": "Config file not found"}), 404

        with open(config_path, "r") as f:
            config_data = f.read()

        return jsonify({"success": True, "config": config_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/evaluation/checkpoints/<session_id>", methods=["GET"])
def list_checkpoints(session_id):
    """
    List available checkpoints for a session.

    Args:
        session_id: Training session identifier.

    Returns:
        JSON array of checkpoint info (name, path, size, modified time).
    """
    try:
        checkpoints = evaluation_manager.list_checkpoints(session_id)
        return jsonify({"success": True, "checkpoints": checkpoints})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/evaluation/start", methods=["POST"])
def start_evaluation():
    """
    Start model evaluation.

    Expected JSON body:
        checkpoint_path: Path to model checkpoint.
        num_episodes: Number of evaluation episodes.
        record_video: Whether to record videos.
        device: Computing device.

    Returns:
        JSON with eval_id on success.
    """
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        checkpoint_path = data.get("checkpoint_path")
        num_episodes = data.get("num_episodes") or 10
        record_video = data.get("record_video", True)
        device = data.get("device") or "cuda"

        result = evaluation_manager.start_evaluation(
            checkpoint_path, num_episodes, record_video, device
        )
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/evaluation/status", methods=["GET"])
def evaluation_status():
    """
    Get current evaluation status and results.

    Returns:
        JSON with evaluation progress, episode results, and videos.
    """
    status = evaluation_manager.get_status()
    if status is None:
        return jsonify({"is_evaluating": False})
    return jsonify({"is_evaluating": status.get("is_active", False), **status})


@app.route("/api/video/<path:video_path>")
def serve_video(video_path):
    """
    Serve video file.

    Args:
        video_path: Relative path to video file.

    Returns:
        Video file stream with proper MIME type.
    """
    try:
        abs_path = os.path.abspath(video_path)
        if not os.path.exists(abs_path):
            return jsonify({"error": "Video not found"}), 404

        return send_file(abs_path, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON with status and version info.
    """
    return jsonify({
        "status": "healthy",
        "service": "DQN DemonAttack Web Interface",
        "training_active": training_manager.is_training(),
        "evaluation_active": evaluation_manager.is_evaluating()
    })


def main(host="0.0.0.0", port=5000, debug=False):
    """
    Start Flask development server.

    Args:
        host: Host address to bind.
        port: Port number to bind.
        debug: Enable debug mode.
    """
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main(debug=True)
