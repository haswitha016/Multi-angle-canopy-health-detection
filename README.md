# Multi Health Canopy Detection

This project estimates canopy health from RGB images using a practical computer-vision workflow. The current implementation is aligned to the real dataset used in this workspace: it classifies canopy appearance into two classes, **Healthy** and **Stressed**, rather than inventing a three-class mild/severe model.

## Project goal

The app supports a multi-view canopy assessment workflow where several images of the same plant or canopy are analyzed separately and then fused into a single final estimate. The dashboard is designed to present the output in a useful field-analysis format that is easy to understand and demo to a user.

## Current implementation status

The project currently includes:

- Flask dashboard with upload, analysis, and report download
- multi-angle image upload and view labeling
- confidence-weighted fusion pipeline
- health score calculation on a 0–100 scale
- Grad-CAM generation for trained-model predictions
- image quality analysis for blur, brightness, resolution, and vegetation coverage
- responsive dashboard UI with result cards, charts, and history
- practical care recommendations such as watering, soil checks, and monitoring

The model and UI are intentionally kept consistent with the real dataset labels: **Healthy** and **Stressed**.

## Supported classes

- Healthy
- Stressed

The project does not claim disease diagnosis or a mild/severe stage that is not supported by the training data.

## File overview

```text
canopy_health/
├── app.py                     # Flask app: upload route, analysis API, report download
├── README.md                 # Usage and project documentation
├── PROJECT_REPORT.md         # Short project overview for presentation/demo use
├── data/                     # Local dataset directory
├── explainability/           # Grad-CAM logic
├── inference/
│   ├── fusion.py             # Fusion and health score logic
│   └── predictor.py          # Demo mode + trained inference wrapper
├── models/                   # Model definitions and checkpoints
├── preprocessing/            # Image quality checks and vegetation estimates
├── reports/                  # Downloadable JSON reports
├── static/
│   ├── css/style.css         # Dashboard styling
│   ├── js/app.js            # Front-end interactions and charts
│   └── uploads/              # Uploaded files and generated Grad-CAM overlays
├── templates/index.html      # Dashboard page
├── tests/
│   ├── test_app.py           # API-level smoke tests
│   └── test_fusion.py        # Fusion logic tests
├── training/
│   ├── dataset.py            # Data loading and transforms
│   └── train.py              # Training workflow
├── requirements.txt          # Python dependencies
├── .env.example              # Example local environment settings
└── prepare_dataset.py        # Data-prep helper
```

## Environment setup

Use the workspace virtual environment in this project or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```bash
cd /home/haswitha/projects/canopy_health
./.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Demo usage

1. Upload one or more canopy images.
2. Give each image a label such as Front, Left, Right, or Back.
3. Select the mode: Demo mode or Trained model.
4. Click Analyze image set.
5. Review the fused health estimate, confidence score, care recommendation, and Grad-CAM output.

## Health interpretation

The fused health score is a 0–100 indicator based on class probabilities:

- Higher score = healthier canopy estimate
- Lower score = more stressed canopy estimate

This is a project-defined visual estimate and not a medically or agronomically validated diagnosis.

## Care guidance logic

The app recommends practical actions such as:

- continue the current watering routine when the canopy appears healthy
- check soil moisture and re-capture images if the image quality is poor
- water more consistently and inspect soil conditions when the canopy appears stressed

These suggestions are operational guidance and not a plant-health diagnosis.

## Validation

The project includes automated checks for:

- app API behavior
- fusion correctness
- confidence weighting
- quality weighting fallback for zero weights
- invalid image rejection

Verified command:

```bash
cd /home/haswitha/projects/canopy_health && ./.venv/bin/python -m unittest discover -s tests -v && node --check static/js/app.js
```

## Notes

- The current dataset and labels are binary, so the app intentionally uses Healthy and Stressed only.
- The demo mode is a heuristic preview and should not be used as a trained model result.
- The trained-model path requires a real checkpoint and should be used only when the corresponding checkpoint exists.
