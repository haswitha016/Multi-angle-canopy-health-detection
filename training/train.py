"""Train and evaluate a MobileNetV3-Small canopy stress classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

from inference.fusion import CLASS_NAMES
from models.network import build_model
from training.dataset import build_loaders


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, labels_all, predictions_all = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, predictions_all, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            labels_all, predictions_all, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }


def train(dataset_dir: Path, model_path: Path, epochs: int, patience: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(dataset_dir, batch_size=batch_size)
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1, stale_epochs = -1.0, 0
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        training_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_metrics = run_epoch(model, loaders["validation"], criterion, device)
        print(
            f"Epoch {epoch:03d}/{epochs} | train loss {training_metrics['loss']:.4f}, "
            f"acc {training_metrics['accuracy']:.3f}, P/R/F1 "
            f"{training_metrics['precision']:.3f}/{training_metrics['recall']:.3f}/{training_metrics['f1']:.3f} "
            f"| val loss {validation_metrics['loss']:.4f}, acc {validation_metrics['accuracy']:.3f}, "
            f"P/R/F1 {validation_metrics['precision']:.3f}/{validation_metrics['recall']:.3f}/"
            f"{validation_metrics['f1']:.3f}"
        )
        if validation_metrics["f1"] > best_f1:
            best_f1, stale_epochs = validation_metrics["f1"], 0
            torch.save(
                {"state_dict": model.state_dict(), "class_names": list(CLASS_NAMES), "image_size": 224},
                model_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    print(f"Test metrics (macro precision/recall/F1; confusion matrix order: {', '.join(CLASS_NAMES)}):")
    print(test_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/canopy_health_best.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.patience, args.batch_size)


if __name__ == "__main__":
    main()
