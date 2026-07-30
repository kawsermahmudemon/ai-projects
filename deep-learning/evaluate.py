"""
🧠 Deep Learning — Evaluate Model
====================================
Load a trained MNIST CNN and compute detailed evaluation metrics:
test accuracy, confusion matrix, per-class precision/recall, and error analysis.

Usage:
    python evaluate.py
    python evaluate.py --model-dir models
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model as keras_load_model


def load_test_data():
    """Load and preprocess MNIST test data."""
    print("📥 Loading MNIST test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    print(f"   Test samples: {x_test.shape[0]:,}\n")
    return x_test, y_test


def plot_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — MNIST CNN", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()

    return cm


def plot_error_examples(x_test, y_true, y_pred, save_path: str = "error_examples.png"):
    """Show examples of misclassified digits."""
    errors = np.where(y_true != y_pred)[0]

    if len(errors) == 0:
        print("🎉 No errors found — perfect accuracy!")
        return

    # Show up to 20 errors
    n_show = min(20, len(errors))
    fig, axes = plt.subplots(2, 10, figsize=(16, 4))

    for i, ax in enumerate(axes.flat):
        if i < n_show:
            idx = errors[i]
            ax.imshow(x_test[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"P:{y_pred[idx]} T:{y_true[idx]}", color="red", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Misclassified Digits ({len(errors)} total errors)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"🖼️  Error examples saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="🧠 Evaluate MNIST CNN")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with trained model")
    args = parser.parse_args()

    print("=" * 55)
    print("  🧠 Deep Learning — Model Evaluation")
    print("=" * 55 + "\n")

    # Load model
    model_path = os.path.join(args.model_dir, "mnist_cnn.keras")
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at: {model_path}")
        print("   Run 'python train.py' first.")
        sys.exit(1)

    print(f"🔄 Loading model from: {model_path}")
    model = keras_load_model(model_path)
    print("✅ Model loaded!\n")

    # Load test data
    x_test, y_true = load_test_data()

    # Evaluate
    print("📏 Running evaluation on test set...\n")
    y_pred_probs = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    total_errors = np.sum(y_pred != y_true)
    print(f"\n{'=' * 50}")
    print(f"  ✅ Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"  ❌ Total Errors:   {total_errors} / {len(y_true)}")
    print(f"{'=' * 50}\n")

    # Per-class classification report
    print("📋 Per-Class Classification Report:\n")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    # Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred)

    # Error analysis
    plot_error_examples(x_test, y_true, y_pred)

    # Most confused pairs
    print("\n🔍 Most Confused Digit Pairs:")
    np.fill_diagonal(cm, 0)
    for _ in range(5):
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        if cm[i, j] == 0:
            break
        print(f"   {i} → {j}: {cm[i, j]} misclassifications")
        cm[i, j] = 0

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
