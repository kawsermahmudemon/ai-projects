"""
📊 Machine Learning — Utility Functions
=========================================
Shared helpers for data loading, visualization, and evaluation metrics.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


def plot_confusion_matrix(
    y_true, y_pred, labels: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class label names.
        title: Plot title.
        save_path: Path to save the figure.
    """
    if plt is None:
        print("⚠️  matplotlib not installed. Skipping plot.")
        return

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()


def plot_feature_importance(
    feature_names: list, importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: str = None,
):
    """
    Plot a horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values.
        title: Plot title.
        top_n: Number of top features to show.
        save_path: Path to save the figure.
    """
    if plt is None:
        print("⚠️  matplotlib not installed. Skipping plot.")
        return

    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    ax.barh(sorted_names, sorted_importances, color=colors, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Feature importance saved to: {save_path}")
    plt.show()


def plot_model_comparison(
    model_names: list, accuracies: list,
    title: str = "Model Comparison",
    save_path: str = None,
):
    """
    Plot a bar chart comparing model accuracies.

    Args:
        model_names: List of model names.
        accuracies: List of accuracy scores.
        title: Plot title.
        save_path: Path to save the figure.
    """
    if plt is None:
        print("⚠️  matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, [a * 100 for a in accuracies], color=colors, edgecolor="white", width=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc * 100:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Model comparison saved to: {save_path}")
    plt.show()


def print_classification_results(y_true, y_pred, target_names: list = None):
    """Print classification report and accuracy."""
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  ✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 50}\n")

    print("📋 Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

    return accuracy
