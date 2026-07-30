"""
👁️ Computer Vision — Image Classifier
========================================
Classify images using a pre-trained MobileNetV2 model (ImageNet, 1000 classes) via PyTorch.
Runs on CPU — no GPU required.
"""

import argparse
import os
import sys
import urllib.request

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utils import (
    preprocess_image,
    decode_predictions_formatted,
    display_predictions,
    print_predictions,
)


def load_model():
    """Load the pre-trained MobileNetV2 model."""
    print("🔄 Loading PyTorch MobileNetV2 model (pre-trained on ImageNet)...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()  # Set model to evaluation mode
    categories = weights.meta["categories"]
    print("✅ Model loaded successfully!\n")
    return model, categories


def classify_image(model, categories, image_path: str, top: int = 5) -> list[dict]:
    """
    Classify an image and return top predictions.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    print(f"🖼️  Classifying: {image_path}")
    input_batch = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_batch)
    
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    results = decode_predictions_formatted(probabilities, categories, top=top)
    return results


def download_sample_image() -> str:
    """Download a sample image if none is provided."""
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, "sample_dog.jpg")

    if not os.path.exists(sample_path):
        print("📥 No image provided. Downloading a sample image...")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(sample_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"✅ Sample image saved to: {sample_path}\n")
        except Exception as e:
            print(f"❌ Failed to download sample image: {e}")
            sys.exit(1)
    return sample_path


def main():
    parser = argparse.ArgumentParser(
        description="🖼️ Image Classifier — MobileNetV2 (ImageNet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", default=None, help="Path to image file")
    parser.add_argument("--top", type=int, default=5, help="Number of top predictions (default: 5)")
    parser.add_argument("--no-plot", action="store_true", help="Disable graphical output")
    args = parser.parse_args()

    print("=" * 55)
    print("  👁️  Computer Vision — Image Classifier (PyTorch)")
    print("  📦 Model: MobileNetV2 (ImageNet, 1000 classes)")
    print("=" * 55 + "\n")

    model, categories = load_model()

    image_path = args.image if args.image else download_sample_image()
    predictions = classify_image(model, categories, image_path, top=args.top)

    print_predictions(predictions)

    if not args.no_plot:
        try:
            display_predictions(image_path, predictions)
        except Exception as e:
            print(f"⚠️  Could not display plot: {e}")

if __name__ == "__main__":
    main()
