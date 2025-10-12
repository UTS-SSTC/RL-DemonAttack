"""
Flask web application for DQN training and evaluation GUI.

Provides REST API endpoints for training management, model evaluation,
video playback, and metrics visualization.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from dqn_demon_attack.web.training_manager import TrainingManager, TrainingConfig
from dqn_demon_attack.web.evaluation_manager import EvaluationManager


app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))
CORS(app)

training_manager = TrainingManager(base_dir="runs")
evaluation_manager = EvaluationManager(base_dir="runs")


@app.route("/")
def index():
    """Render main GUI page."""
    return render_template("index.html")


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """
    Start a new training session.

    Expected JSON body:
        exp_name: str
        total_steps: int
        eval_every: int
        Other training config parameters

    Returns:
        JSON with session_id and run_dir.
    """
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        config = TrainingConfig(**data)
        result = training_manager.start_training(config)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """Stop current training session."""
    try:
        training_manager.stop_training()
        return jsonify({"success": True, "message": "Training stop requested"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """Get current training status and metrics."""
    status = training_manager.get_status()
    if status is None:
        return jsonify({"is_training": False})

    safe_status = dict(status)
    if "config" in safe_status:
        del safe_status["config"]

    return jsonify({"is_training": True, **safe_status})


@app.route("/api/training/curves/<session_id>", methods=["GET"])
def training_curves(session_id):
    """
    Get training curves for visualization.

    Args:
        session_id: Training session identifier.

    Returns:
        JSON with metrics arrays.
    """
    try:
        curves = training_manager.get_training_curves(session_id)
        return jsonify({"success": True, "data": curves})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List all training sessions."""
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


@app.route("/api/evaluation/checkpoints/<session_id>", methods=["GET"])
def list_checkpoints(session_id):
    """
    List available checkpoints for a session.

    Args:
        session_id: Training session identifier.

    Returns:
        JSON with checkpoint list.
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
        checkpoint_path: str
        num_episodes: int
        record_video: bool
        device: str

    Returns:
        JSON with eval_id.
    """
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        checkpoint_path = data.get("checkpoint_path")
        num_episodes = data.get("num_episodes", 10)
        record_video = data.get("record_video", True)
        device = data.get("device", "cuda")

        result = evaluation_manager.start_evaluation(
            checkpoint_path, num_episodes, record_video, device
        )
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/evaluation/status", methods=["GET"])
def evaluation_status():
    """Get current evaluation status and results."""
    status = evaluation_manager.get_status()
    if status is None:
        return jsonify({"is_evaluating": False})
    return jsonify({"is_evaluating": True, **status})


@app.route("/api/video/<path:video_path>")
def serve_video(video_path):
    """
    Serve video file.

    Args:
        video_path: Relative path to video file.

    Returns:
        Video file stream.
    """
    try:
        abs_path = os.path.abspath(video_path)
        if not os.path.exists(abs_path):
            return jsonify({"error": "Video not found"}), 404

        return send_file(abs_path, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/static/<path:filepath>")
def serve_static(filepath):
    """Serve static files."""
    return send_from_directory(os.path.dirname(os.path.abspath(filepath)),
                               os.path.basename(filepath))


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
