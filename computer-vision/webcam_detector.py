"""
👁️ Computer Vision — YOLOv8 Real-Time Object Detector
======================================================
State-of-the-art real-time object detection using YOLOv8.
"""

import sys
import cv2
from ultralytics import YOLO

def main():
    print("=" * 55)
    print("  📷 Real-Time Webcam YOLO11 Object Detector")
    print("  📦 Model: YOLO11 Small (yolo11s.pt - Fast & Accurate)")
    print("  🎮 Controls: 'q' = Quit")
    print("=" * 55 + "\n")

    print("🔄 Loading YOLO11 Small model (optimized for speed)...")
    try:
        model = YOLO('yolo11s.pt')
    except Exception as e:
        print(f"❌ Failed to load YOLO11 model: {e}")
        sys.exit(1)
    print("✅ Model loaded!\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Make sure a camera is connected.")
        sys.exit(1)

    print("📷 Webcam opened. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO11 inference on the frame (imgsz=480 for balance, conf=0.15 to detect difficult objects)
        results = model.predict(frame, imgsz=480, conf=0.15, verbose=False)

        # Ultralytics natively draws beautiful bounding boxes, labels, and probabilities
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO11 Real-Time Object Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
