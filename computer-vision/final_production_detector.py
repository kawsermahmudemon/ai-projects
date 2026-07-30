"""
=============================================================================
🤖 Supreme AI Object Detector (Final Production Release)
=============================================================================
Author: AI Assistant & User
Description: High-performance, real-time object detection using Ultralytics YOLO.
Features: 
- 600+ classes (OpenImages V7).
- Fullscreen 720p HD Support.
- Custom High-Tech Sci-Fi UI and Bounding Boxes.
- Real-time status panel.
=============================================================================
"""

import sys
import time
import os
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_NAME = 'yolov8s-oiv7.pt'
INFERENCE_RESOLUTION = 640
CONFIDENCE_THRESHOLD = 0.15
# Colors (B, G, R)
PRIMARY_COLOR = (255, 200, 0)   # Cyan/Blue
ACCENT_COLOR = (0, 255, 0)      # Green
ALERT_COLOR = (0, 0, 255)       # Red
BG_COLOR = (15, 15, 15)
# ---------------------

def draw_corner_rect(img, x1, y1, x2, y2, color, length=20, thickness=3):
    """Draws a high-tech bounding box with corner brackets."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    
    # Draw faint full rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def draw_sidebar(frame, fps, is_recording, detected_items):
    """Draws a professional, sci-fi transparent sidebar."""
    h, w = frame.shape[:2]
    sidebar_width = 300
    
    # Semi-transparent overlay for sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_width, 0), (w, h), BG_COLOR, -1)
    
    # Top Bar overlay
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    
    # Bottom Bar overlay
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    
    # Apply alpha blending
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # --- TOP BAR ---
    cv2.putText(frame, " SUPREME AI DETECTOR (PRO)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRIMARY_COLOR, 2)
                
    # Status Indicator (Glowing Green dot)
    pulse = int((np.sin(time.time() * 5) + 1) * 127) # Pulse effect
    cv2.circle(frame, (w - sidebar_width + 30, 20), 6, (0, pulse, 0), -1)
    cv2.circle(frame, (w - sidebar_width + 30, 20), 4, (0, 255, 0), -1)
    cv2.putText(frame, "SYSTEM ONLINE", (w - sidebar_width + 50, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    # --- SIDEBAR CONTENT ---
    start_x = w - sidebar_width + 20
    
    # Title
    cv2.putText(frame, "DASHBOARD", (start_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (start_x, 90), (w - 20, 90), PRIMARY_COLOR, 1)
    
    # Model Info
    cv2.putText(frame, "MODEL:", (start_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "OpenImages V7", (start_x + 80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRIMARY_COLOR, 2)
    cv2.putText(frame, "CAPACITY: 600 Objects", (start_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # FPS Gauge
    cv2.putText(frame, f"FPS: {fps:.1f}", (start_x, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # Draw simple progress bar for FPS
    bar_width = 200
    fill = min(int((fps / 30.0) * bar_width), bar_width)
    cv2.rectangle(frame, (start_x, 220), (start_x + bar_width, 230), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, 220), (start_x + fill, 230), ACCENT_COLOR if fps > 15 else ALERT_COLOR, -1)
    
    # Detected Items List
    cv2.line(frame, (start_x, 270), (w - 20, 270), PRIMARY_COLOR, 1)
    cv2.putText(frame, "LIVE DETECTIONS:", (start_x, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y_offset = 330
    for item, conf in list(detected_items.items())[:10]: # Max 10 items
        # Draw small square icon
        cv2.rectangle(frame, (start_x, y_offset - 10), (start_x + 10, y_offset), PRIMARY_COLOR, -1)
        cv2.putText(frame, f"{item.upper()} ({int(conf*100)}%)", (start_x + 25, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)
        y_offset += 30

    # Recording Status
    if is_recording:
        cv2.rectangle(frame, (start_x, h - 130), (w - 20, h - 90), (0, 0, 50), -1)
        cv2.circle(frame, (start_x + 30, h - 110), 8, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING...", (start_x + 55, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(frame, (start_x, h - 130), (w - 20, h - 90), ALERT_COLOR, 2)
        
    # --- BOTTOM BAR ---
    controls = "[Q] UIT   |   [S] AVE PHOTO   |   [R] ECORD VIDEO"
    cv2.putText(frame, controls, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
    return frame

def main():
    print("=" * 70)
    print(" 🌟 LAUNCHING FULLSCREEN SUPREME DETECTOR ")
    print("=" * 70)
    
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(0)
    # Force high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Cannot open webcam.")
        sys.exit(1)

    # Set up Fullscreen Window
    window_name = "Supreme AI Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_time = time.time()
    fps = 0.0
    frame_count = 0
    is_recording = False
    video_writer = None
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Horizontal flip for mirror effect (more natural)
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 5 == 0:
            elapsed = time.time() - fps_time
            fps = 5.0 / elapsed if elapsed > 0 else 0
            fps_time = time.time()

        # Run Inference
        results = model.predict(frame, imgsz=INFERENCE_RESOLUTION, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detected_items = {}
        
        # Custom Bounding Box Drawing
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            
            # Save highest confidence for sidebar
            if name not in detected_items or conf > detected_items[name]:
                detected_items[name] = conf
                
            # Draw custom sci-fi box
            draw_corner_rect(frame, x1, y1, x2, y2, PRIMARY_COLOR, length=25, thickness=2)
            
            # Draw label background
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 10, y1), PRIMARY_COLOR, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw UI
        final_frame = draw_sidebar(frame, fps, is_recording, detected_items)

        # Handle Recording
        if is_recording:
            if video_writer is None:
                h, w = final_frame.shape[:2]
                filename = f"outputs/record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            video_writer.write(final_frame)

        cv2.imshow(window_name, final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            filename = f"outputs/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, final_frame)
            
        elif key == ord('r') or key == ord('R'):
            is_recording = not is_recording
            if not is_recording and video_writer is not None:
                video_writer.release()
                video_writer = None

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
