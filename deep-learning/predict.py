"""
🧠 Deep Learning — Predict Digits
====================================
Load a trained MNIST CNN and predict digits from images.
Includes an interactive drawing mode using OpenCV.

Usage:
    python predict.py                              # Interactive drawing canvas
    python predict.py --image path/to/digit.png    # Predict from an image file
"""

import argparse
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model as keras_load_model


def load_trained_model(model_dir: str = "models") -> object:
    """Load the trained MNIST CNN model."""
    model_path = os.path.join(model_dir, "mnist_cnn.keras")
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at: {model_path}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    print(f"🔄 Loading model from: {model_path}")
    model = keras_load_model(model_path)
    print("✅ Model loaded!\n")
    return model


def predict_from_image(model, image_path: str) -> tuple[int, float]:
    """
    Predict a digit from an image file.

    Args:
        model: Trained Keras model.
        image_path: Path to a digit image (any size, will be resized).

    Returns:
        Tuple of (predicted_digit, confidence).
    """
    from PIL import Image

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Load and preprocess
    img = Image.open(image_path).convert("L")  # Grayscale
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0

    # MNIST expects white digits on black background
    # If image is mostly white, invert
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    predicted_digit = int(np.argmax(preds))
    confidence = float(preds[predicted_digit])

    return predicted_digit, confidence


def interactive_canvas(model):
    """
    Open an interactive canvas where the user can draw a digit with the mouse.
    Press 'p' to predict, 'c' to clear, 'q' to quit.
    """
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV is required for interactive mode.")
        print("   Install it: pip install opencv-python")
        print("   Or use: python predict.py --image <path>")
        sys.exit(1)

    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    drawing = False
    last_x, last_y = -1, -1

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, (last_x, last_y), (x, y), 255, thickness=20)
            last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    window_name = "Draw a Digit — 'p'=Predict | 'c'=Clear | 'q'=Quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw)

    print("🎨 Interactive Canvas Opened!")
    print("   Draw a digit (0-9) with your mouse.")
    print("   'p' = Predict | 'c' = Clear | 'q' = Quit\n")

    while True:
        # Display canvas with instructions
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, "'p'=Predict  'c'=Clear  'q'=Quit",
                    (10, canvas_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            print("🧹 Canvas cleared.")
        elif key == ord("p"):
            # Resize canvas to 28x28 and predict
            digit_img = cv2.resize(canvas, (28, 28)).astype("float32") / 255.0
            digit_img = digit_img.reshape(1, 28, 28, 1)

            preds = model.predict(digit_img, verbose=0)[0]
            predicted = int(np.argmax(preds))
            confidence = float(preds[predicted])

            print(f"🔮 Predicted: {predicted}  (Confidence: {confidence * 100:.1f}%)")

            # Show all probabilities
            for i in range(10):
                bar = "█" * int(preds[i] * 20) + "░" * (20 - int(preds[i] * 20))
                marker = " ◀" if i == predicted else ""
                print(f"   {i}: {bar} {preds[i]*100:5.1f}%{marker}")
            print()

    cv2.destroyAllWindows()
    print("👋 Canvas closed.")


def main():
    parser = argparse.ArgumentParser(description="🧠 Predict Handwritten Digits")
    parser.add_argument("--image", type=str, default=None, help="Path to a digit image file")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with trained model")
    args = parser.parse_args()

    print("=" * 55)
    print("  🧠 Deep Learning — Digit Prediction")
    print("=" * 55 + "\n")

    model = load_trained_model(args.model_dir)

    if args.image:
        digit, confidence = predict_from_image(model, args.image)
        print(f"🔮 Predicted Digit: {digit}")
        print(f"   Confidence:      {confidence * 100:.1f}%")
    else:
        interactive_canvas(model)


if __name__ == "__main__":
    main()
