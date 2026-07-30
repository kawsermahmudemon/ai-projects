"""
👁️ Computer Vision — Utility Functions
========================================
Shared helpers for image preprocessing, visualization, and result formatting.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def preprocess_image(image_path: str):
    """
    Load and preprocess an image for MobileNetV2 inference using PyTorch.
    """
    from PIL import Image
    from torchvision import transforms
    
    img = Image.open(image_path).convert("RGB")
    
    # Standard PyTorch ImageNet transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch


def decode_predictions_formatted(probs, categories, top: int = 5) -> list[dict]:
    """
    Decode model predictions into a human-readable list.
    probs: 1D tensor of probabilities
    categories: list of category names
    """
    import torch
    
    top_prob, top_catid = torch.topk(probs, top)
    
    results = []
    for i in range(top_prob.size(0)):
        results.append({
            "class_id": top_catid[i].item(),
            "label": categories[top_catid[i].item()].replace("_", " ").title(),
            "confidence": float(top_prob[i].item()),
        })
    return results


def display_predictions(image_path: str, predictions: list[dict]):
    """
    Display the image alongside a bar chart of top predictions.
    """
    if plt is None:
        print("⚠️  matplotlib not installed. Showing text-only results.")
        print_predictions(predictions)
        return

    from PIL import Image

    img = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Show image
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Input Image", fontsize=14, fontweight="bold")

    # Show predictions as horizontal bar chart
    labels = [p["label"] for p in reversed(predictions)]
    confidences = [p["confidence"] * 100 for p in reversed(predictions)]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))

    bars = ax2.barh(labels, confidences, color=colors, edgecolor="white", height=0.6)
    ax2.set_xlabel("Confidence (%)", fontsize=12)
    ax2.set_title("Top Predictions", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 100)

    # Add percentage labels on bars
    for bar, conf in zip(bars, confidences):
        ax2.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%", va="center", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    print("📊 Result saved to prediction_result.png")
    plt.show()


def print_predictions(predictions: list[dict]):
    """Print predictions to the console in a formatted table."""
    print("\n" + "=" * 50)
    print("  🏷️  Top Predictions")
    print("=" * 50)
    for i, pred in enumerate(predictions, 1):
        bar_length = int(pred["confidence"] * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {pred['label']:<25} {bar} {pred['confidence']*100:.1f}%")
    print("=" * 50 + "\n")
