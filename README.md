# Multi-Angle Canopy Health Estimation

A staged academic computer-vision project to estimate the health of the same canopy from multiple RGB views. The intended output classes are **Healthy**, **Mild Stress**, and **Severe Stress**. The project estimates visible stress categories; it does not diagnose a specific disease.

## Current status: Stage 1 of 13

This first stage creates the project structure and environment setup. Model training, inference, multi-angle fusion, image-quality analysis, Grad-CAM, and the Flask dashboard are intentionally not implemented yet. Later stages will add those components incrementally.

## Project structure

```text
canopy_health/
├── data/              # Local dataset location; not committed
├── explainability/    # Grad-CAM implementation (later stage)
├── inference/         # Single- and multi-image prediction (later stages)
├── models/            # Model definitions and local checkpoints
├── preprocessing/     # Image transforms and vegetation segmentation
├── reports/           # Generated analysis reports; not committed
├── static/
│   ├── css/            # Dashboard styles
│   ├── js/             # Dashboard interactions and charts
│   └── uploads/        # Temporary uploaded images; not committed
├── templates/         # Flask/Jinja HTML templates
├── training/          # Training and evaluation scripts
├── utils/             # Shared configuration and helper functions
├── .env.example       # Example local configuration
├── app.py             # Flask entry point (added in the backend stage)
└── requirements.txt   # Python dependencies
```

Python package markers (`__init__.py`) will be added to code directories as their modules are implemented. The `.gitkeep` files preserve currently empty folders in version control.

## Environment setup

Use Python 3.10 or newer. From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
# Multi-Angle Canopy Health Estimation

A beginner-friendly academic computer-vision project that estimates visible canopy stress from several RGB photographs of the **same plant or canopy**. It uses three broad visual categories: **Healthy**, **Mild Stress**, and **Severe Stress**. It does not diagnose a particular disease.

## What is implemented

- CPU- or CUDA-capable MobileNetV3-Small transfer-learning training and test evaluation.
- Configurable `train/`, `validation/`, and `test/` dataset loading with realistic training augmentation.
- Per-image class probabilities, confidence, expected health score, and image-quality measures.
- Majority-vote, average-probability, and confidence-weighted multi-view fusion.
- Optional quality-adjusted confidence weighting.
- Grad-CAM overlays for checkpoint-backed model predictions.
- Flask upload/API and responsive dashboard with charts and downloadable JSON reports.
- An explicitly labelled **DEMO MODE** for trying the dashboard without a trained checkpoint.

Demo estimates are deterministic heuristics based on image brightness and detected green coverage. They are not ML predictions, evidence of plant health, or model validation. Choose **Trained model** only after training a checkpoint.

## Architecture and important files

```text
canopy_health/
├── app.py                       # Flask routes, uploads, model selection, report downloads
├── data/                         # Local dataset root; kept out of Git
├── explainability/gradcam.py     # Grad-CAM overlay generation
├── inference/fusion.py           # Fusion rules and health-score mapping
├── inference/predictor.py        # Trained inference and separate demo heuristic
├── models/network.py             # MobileNetV3-Small classifier
├── preprocessing/quality.py      # HSV/ExG mask and image-quality metrics
├── reports/                      # Generated JSON reports; kept out of Git
├── static/css/style.css          # Dashboard styles
├── static/js/app.js              # Upload workflow, results, and charts
├── static/uploads/               # Uploaded originals and Grad-CAM images
├── templates/index.html          # Dashboard markup
├── tests/test_fusion.py          # Fusion and health-score unit tests
├── training/dataset.py           # Dataset checks and PyTorch transforms
├── training/train.py             # Training, early stopping, and test metrics
├── .env.example                  # Local path and Flask settings
└── requirements.txt              # Python dependencies
```

## Environment installation

Use Python 3.10 or newer. At the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The project selects CUDA automatically when available and otherwise uses CPU. To guarantee CPU-only PyTorch wheels on Linux, install them from the CPU wheel index before the rest of the requirements:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

For CUDA, select the matching PyTorch/torchvision command at <https://pytorch.org/get-started/locally/> instead. PyTorch may download pretrained MobileNetV3 weights the first time training is run with network access.

Verify the environment:

```bash
python -c "import torch, torchvision, cv2, flask; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| OpenCV', cv2.__version__, '| Flask', flask.__version__)"
```

## Dataset format

Set `DATASET_DIR` in `.env`, or pass `--dataset` to training. Class directory names and capitalization are checked:

```text
<DATASET_DIR>/
├── train/{Healthy,Mild Stress,Severe Stress}/
├── validation/{Healthy,Mild Stress,Severe Stress}/
└── test/{Healthy,Mild Stress,Severe Stress}/
```

Keep every image of a physical plant/canopy in only one split. Otherwise, near-identical views may leak between training and test and make evaluation misleading. Use representative lighting, backgrounds, camera distances, cultivars, and stress levels, with labels reviewed by a knowledgeable person. Do not commit a dataset unless its license permits it.

## Preprocessing

Images are converted to RGB, resized/cropped to 224×224, converted to tensors, and normalized with the ImageNet mean and standard deviation used by pretrained MobileNetV3. Training adds random resized crops, horizontal flips, rotations, and mild brightness/contrast/saturation/hue jitter. Validation and test transforms are deterministic.

The optional vegetation estimate combines an HSV green range and excess-green (`ExG = 2G - R - B`) threshold. It is quality/demo context, not a biological segmentation label or proof of canopy boundaries.

## Training and evaluation

With the virtual environment active and the dataset in place:

```bash
python -m training.train --dataset ./data --output ./models/canopy_health_best.pth --epochs 30 --patience 5 --batch-size 16
```

The script trains MobileNetV3-Small with a three-output classification head, AdamW, and cross-entropy. It saves the checkpoint with the best validation macro-F1 and stops after the configured number of epochs without improvement. It reloads that checkpoint and prints test accuracy, macro precision, macro recall, macro F1, and a confusion matrix in Healthy/Mild/Severe order. `--dataset`, `--output`, `--epochs`, `--patience`, and `--batch-size` are configurable.

Use the test set once for final evaluation, not repeatedly to tune model choices. For a formal study, report class support and uncertainty, particularly when classes are imbalanced.

## Run the dashboard

```bash
python app.py
```

Open <http://127.0.0.1:5000>. Upload up to 12 images of one canopy, optionally name each view, select **DEMO MODE** or **Trained model**, and analyze. Demo mode works without a checkpoint after installing runtime dependencies. Trained mode uses `MODEL_PATH` (default `./models/canopy_health_best.pth`) and returns an error if the checkpoint is absent. Uploaded images and Grad-CAM images are kept under `UPLOAD_DIR`; JSON reports are under `REPORT_DIR`. Paths, host, port, and upload limit can be configured in `.env`.

Uploads are saved locally and are not automatically deleted. Do not expose Flask's development server publicly; add authentication, secure storage, retention limits, and deployment hardening before wider use.

## Multi-angle fusion

Each view gives a probability vector over the same three classes. The dashboard compares:

- **Majority vote:** count each view's most likely class and normalize the vote counts. Ties follow the fixed class order.
- **Average probability:** take the arithmetic mean of each class probability across views.
- **Confidence-weighted:** weight each view by its maximum class probability (confidence), then normalize:

    `final_probability[c] = sum(probability[i][c] * confidence[i]) / sum(confidence[i])`

By default, this is exactly confidence-only weighting. The optional dashboard checkbox multiplies each confidence weight by that view's image-quality score. If all adjusted weights are zero, it falls back to average probabilities. Quality is a heuristic, not a calibrated reliability estimate.

## Health score

For fused probabilities:

`score = 100 * P(Healthy) + 50 * P(Mild Stress) + 0 * P(Severe Stress)`

The score is 0–100. Display categories are Healthy for scores ≥ 75, Mild Stress for 25 to < 75, and Severe Stress below 25. Values and thresholds are transparent project conventions, not a validated agronomic scale. The selected fusion method's maximum class probability is shown as confidence.

## Image quality

Per-view fields include Laplacian variance as a sharpness/blur indicator, mean HSV value as brightness, pixel dimensions, and estimated green coverage. Thresholds flag blur, very dark/overexposed images, low resolution, or little detected vegetation. Unusual species, lighting, and backgrounds can fool these heuristics; inspect the source images.

## Grad-CAM

For trained-model predictions, Grad-CAM uses gradients and activations from MobileNetV3's final feature block to create an overlay next to the original image. It highlights image regions contributing to the selected model output. It does **not** prove biological correctness, identify a disease, or establish causal evidence. Demo predictions have no Grad-CAM because no learned model is involved.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover normalization, the confidence-weighted formula, optional quality adjustment, the all-zero quality fallback, and invalid empty input.

## Limitations and future work

- Results depend on representative data, reviewed labels, and careful split design. Without appropriate held-out images, model performance is unknown.
- Softmax confidence is not automatically calibrated and is not the probability that a plant is truly healthy or stressed.
- The score and quality heuristics are unvalidated choices.
- The application assumes uploaded images show the same canopy; it does not verify subject identity or view angle.
- Grad-CAM is qualitative, not a correctness guarantee. Demo mode is a UI preview and must never be reported as an ML result.
- Future work: plant-level split manifests, repeated-seed evaluation and confidence intervals, probability calibration, explicit view metadata, expert-validated scoring, improved segmentation, authenticated deployment, and more diverse data.
