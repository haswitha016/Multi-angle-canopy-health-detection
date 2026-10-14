"""Flask dashboard for multi-angle canopy health estimation."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from inference.fusion import CLASS_NAMES, fuse_predictions
from inference.predictor import CanopyPredictor, demo_predict


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def configured_path(name: str, default: str) -> Path:
    configured = Path(os.getenv(name, default))
    return configured if configured.is_absolute() else ROOT / configured


UPLOAD_DIR = configured_path("UPLOAD_DIR", "static/uploads")
REPORT_DIR = configured_path("REPORT_DIR", "reports")
MODEL_PATH = configured_path("MODEL_PATH", "models/canopy_health_best.pth")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["REPORT_FOLDER"] = REPORT_DIR
_predictor: CanopyPredictor | None = None


def get_predictor() -> CanopyPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint found at {MODEL_PATH}. Train the model first or select DEMO MODE."
            )
        _predictor = CanopyPredictor(MODEL_PATH)
    return _predictor


def build_care_recommendation(predicted_class: str, health_score: float, quality_flags: list[str] | None = None) -> dict[str, str]:
    flags = {flag.lower() for flag in (quality_flags or [])}
    if predicted_class == "Healthy" and health_score >= 60:
        return {
            "title": "Maintain current care",
            "message": "The canopy looks healthy. Continue the current watering routine and monitor soil moisture and foliage condition over time.",
        }
    if any(token in flags for token in {"blurry", "very dark or overexposed", "low resolution", "little detected vegetation"}):
        return {
            "title": "Re-check image quality and soil moisture",
            "message": "The image quality suggests some uncertainty. Re-capture the canopy in clearer light and confirm soil moisture before changing irrigation or soil conditions.",
        }
    if health_score < 35:
        return {
            "title": "Water thoroughly and refresh the soil",
            "message": "The canopy appears stressed. Water the plant more consistently and consider replacing or improving the soil if drainage or nutrient issues are suspected.",
        }
    return {
        "title": "Water and monitor soil moisture",
        "message": "The canopy appears stressed. Increase watering consistency, inspect the soil and root zone, and monitor daily until the plant shows recovery.",
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    files = [file for file in request.files.getlist("images") if file and file.filename]
    if not files:
        return jsonify({"error": "Choose at least one canopy image."}), 400
    if len(files) > 12:
        return jsonify({"error": "Upload up to 12 views per canopy."}), 400

    mode = request.form.get("mode", "demo")
    if mode not in {"demo", "trained_model"}:
        return jsonify({"error": "Unknown prediction mode."}), 400
    if mode == "trained_model":
        try:
            predictor = get_predictor()
        except (FileNotFoundError, RuntimeError) as error:
            return jsonify({"error": str(error)}), 503

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = []
    image_rows = []
    for index, uploaded_file in enumerate(files, start=1):
        original_name = secure_filename(uploaded_file.filename or f"view-{index}.jpg")
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported image type for {original_name}. Use JPG, PNG, WEBP, or BMP."}), 400
        stored_name = f"{uuid.uuid4().hex}{extension}"
        image_path = UPLOAD_DIR / stored_name
        uploaded_file.save(image_path)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError):
            image_path.unlink(missing_ok=True)
            return jsonify({"error": f"Could not decode {original_name} as an image."}), 400
        view_name = request.form.get(f"view_{index}", "").strip() or f"View {index}"
        if mode == "demo":
            result = demo_predict(image_path)
        else:
            cam_name = f"{Path(stored_name).stem}_gradcam.jpg"
            cam_path = UPLOAD_DIR / cam_name
            result = predictor.predict(image_path, str(cam_path))
            result["gradcam_url"] = url_for("uploaded_file", filename=cam_name)
            result.pop("gradcam_path", None)
        predictions.append({**result})
        result.update(
            {
                "view": view_name,
                "filename": original_name,
                "image_url": url_for("uploaded_file", filename=stored_name),
                "probabilities": dict(zip(CLASS_NAMES, result["probabilities"])),
            }
        )
        image_rows.append(result)

    quality_weighted = request.form.get("quality_weighted") == "true"
    fusion_results = fuse_predictions(predictions, quality_weighted=quality_weighted)
    summary = fusion_results["confidence_weighted"]
    quality_flags = []
    for image in image_rows:
        quality_flags.extend(image.get("quality", {}).get("flags", []))
    recommendation = build_care_recommendation(summary["predicted_class"], summary["health_score"], quality_flags)
    report_id = uuid.uuid4().hex
    report = {
        "report_id": report_id,
        "mode": mode,
        "note": "Estimates visible canopy stress categories; not a diagnosis of any specific disease.",
        "images": image_rows,
        "fusion": fusion_results,
        "selected_fusion": "confidence_weighted",
        "quality_weighted": quality_weighted,
        "summary": {
            "predicted_class": summary["predicted_class"],
            "confidence": summary["confidence"],
            "health_score": summary["health_score"],
            "health_category": summary["health_category"],
            "probabilities": summary["probabilities"],
            "recommendation": recommendation,
        },
    }
    (REPORT_DIR / f"{report_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_url"] = url_for("download_report", report_id=report_id)
    return jsonify(report)


@app.get("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/reports/<report_id>")
def download_report(report_id: str):
    if len(report_id) != 32 or any(character not in "0123456789abcdef" for character in report_id):
        return jsonify({"error": "Report not found."}), 404
    path = REPORT_DIR / f"{report_id}.json"
    if not path.is_file():
        return jsonify({"error": "Report not found."}), 404
    return send_from_directory(REPORT_DIR, path.name, as_attachment=True, download_name="canopy-health-report.json")


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Upload is too large. Reduce the total upload size and try again."}), 413


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )