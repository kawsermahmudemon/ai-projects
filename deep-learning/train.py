"""
🧠 Deep Learning — Train MNIST CNN
=====================================
Train a Convolutional Neural Network on the MNIST handwritten digit dataset.
Automatically downloads MNIST, trains the model, saves it, and plots training curves.

Usage:
    python train.py                    # Train with defaults (10 epochs)
    python train.py --epochs 20        # Train for 20 epochs
    python train.py --batch-size 64    # Custom batch size
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from model import build_mnist_cnn


def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess for CNN input.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    print("📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to add channel dimension: (28, 28) → (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"   Training samples: {x_train.shape[0]:,}")
    print(f"   Test samples:     {x_test.shape[0]:,}")
    print(f"   Image shape:      {x_train.shape[1:]}")
    print()

    return x_train, y_train, x_test, y_test


def plot_training_history(history, save_path: str = "training_curves.png"):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history object.
        save_path: Where to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    ax1.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy", markersize=4)
    ax1.plot(epochs, history.history["val_accuracy"], "r-o", label="Validation Accuracy", markersize=4)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.95, 1.0)

    # Loss
    ax2.plot(epochs, history.history["loss"], "b-o", label="Training Loss", markersize=4)
    ax2.plot(epochs, history.history["val_loss"], "r-o", label="Validation Loss", markersize=4)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Training curves saved to: {save_path}")
    plt.show()


def plot_sample_predictions(model, x_test, y_test, save_path: str = "sample_predictions.png"):
    """Show sample predictions on test images."""
    preds = model.predict(x_test[:16], verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test[:16], axis=1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].reshape(28, 28), cmap="gray")
        color = "green" if pred_classes[i] == true_classes[i] else "red"
        ax.set_title(f"P:{pred_classes[i]} T:{true_classes[i]}", color=color, fontsize=10)
        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"🖼️  Sample predictions saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="🧠 Train MNIST CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the trained model")
    args = parser.parse_args()

    print("=" * 55)
    print("  🧠 Deep Learning — MNIST Digit Recognition")
    print("  📦 Architecture: CNN (Conv2D → MaxPool → Dense)")
    print("=" * 55 + "\n")

    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Build model
    print("🔨 Building CNN model...")
    model = build_mnist_cnn()
    model.summary()
    print()

    # Train
    print(f"🏋️ Training for {args.epochs} epochs (batch size: {args.batch_size})...\n")
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
    )

    # Evaluate on test set
    print("\n📏 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"   Test Loss:     {test_loss:.4f}\n")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "mnist_cnn.keras")
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}\n")

    # Plot results
    plot_training_history(history)
    plot_sample_predictions(model, x_test, y_test)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
