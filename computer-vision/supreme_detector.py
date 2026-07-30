"""
👁️ Computer Vision — Supreme OpenImages Object Detector
=========================================================
Detect over 600 different types of everyday objects!
Trained on Google's OpenImages V7 massive dataset.
"""

import sys
import time
from datetime import datetime
import cv2
from ultralytics import YOLO

def main():
    print("=" * 65)
    print("  👑 Supreme Level Object Detector (OpenImages V7)")
    print("  📦 Model: YOLOv8 Small (yolov8s-oiv7.pt - 600 Classes!)")
    print("  🎮 Controls: 'q' = Quit | 's' = Save Frame")
    print("=" * 65 + "\n")

    print("🔄 Loading Supreme OpenImages model (Big Data)...")
    try:
        # Load YOLOv8 model trained on OpenImages V7 (600 classes)
        model = YOLO('yolov8s-oiv7.pt')
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"✅ Model loaded successfully! (Capable of detecting {len(model.names)} objects)\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Make sure a camera is connected.")
        sys.exit(1)

    print("📷 Webcam opened. Press 'q' to quit.\n")

    fps_time = time.time()
    fps = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Calculate FPS
        if frame_count % 15 == 0:
            elapsed = time.time() - fps_time
            fps = 15.0 / elapsed if elapsed > 0 else 0
            fps_time = time.time()

        # Run inference (imgsz=480 for balance of speed and detecting small objects like pens)
        results = model.predict(frame, imgsz=480, conf=0.15, verbose=False)

        # Plot bounding boxes
        annotated_frame = results[0].plot()

        # Draw FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 130, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("👑 Supreme Detector (600 Classes)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Frame saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
