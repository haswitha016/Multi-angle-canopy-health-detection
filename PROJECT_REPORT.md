# Project Report: Multi Health Canopy Detection

## Overview

This project is a canopy health estimation dashboard for analyzing multiple photographs of the same plant or canopy area. It combines visual cues from several views and fuses them into a final health estimate. The current implementation uses the actual dataset structure in this workspace: binary classification into **Healthy** and **Stressed**.

## Main features

- multi-view canopy analysis
- per-image confidence and probability display
- fused health score on a 0–100 scale
- downloadable JSON report
- Grad-CAM explanation for trained-model runs
- image quality assessment for blur, brightness, resolution, and vegetation coverage
- practical guidance such as watering recommendations or soil checks
- modern dashboard interface for demo presentation

## Model and data constraints

The model is intentionally aligned with the current dataset labels:

- Healthy
- Stressed

No mild/severe classes or fabricated disease category are introduced. The project keeps the UI and logic consistent with the actual dataset used in the workspace.

## Technical architecture

- Flask app in `app.py` handles upload and analysis endpoints
- `inference/fusion.py` combines per-view predictions
- `inference/predictor.py` contains demo heuristics and trained-model inference flow
- `preprocessing/quality.py` estimates image quality metrics
- `explainability/gradcam.py` creates Grad-CAM overlays
- `templates/index.html` renders the dashboard
- `static/js/app.js` manages dynamic UI updates and charts
- `static/css/style.css` handles the polished presentation layer

## Practical care guidance

The dashboard suggests practical action based on the combined result and image quality checks. Examples include:

- continue routine watering for a healthy canopy
- check soil moisture and re-capture images if image quality is poor
- water more consistently and inspect soil/root conditions when the canopy is stressed

These are operational recommendations only and are not formal agronomic diagnoses.

## Validation

The project has automated checks covering:

- app route behavior
- fusion calculations
- confidence weighting
- quality-weight fallback
- invalid image rejection

Verified with:

```bash
cd /home/haswitha/projects/canopy_health && ./.venv/bin/python -m unittest discover -s tests -v && node --check static/js/app.js
```

## Demo readiness

The app can be run locally with:

```bash
cd /home/haswitha/projects/canopy_health
./.venv/bin/python app.py
```

Open the app in a browser at:

```text
http://127.0.0.1:5000
```

## Final note

This project is suitable for a presentation demo because it combines a functional analysis workflow with a clear visual dashboard and realistic guidance without overstating the model’s certainty or claiming unsupported disease classes.
