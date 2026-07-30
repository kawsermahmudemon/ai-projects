"""
=============================================================================
🌟 THE ULTIMATE AI VISION SYSTEM (V2.0 MAX) 🌟
=============================================================================
Author: AI Assistant & User
Description: The absolute pinnacle of real-time computer vision.
Added Video Recording and 2 Jaw-Dropping New AI Modes!
[Mode 1] 600-Class Object Detection
[Mode 2] Pixel-Perfect Instance Segmentation
[Mode 3] Human Skeleton Pose Estimation
[Mode 4] Privacy Anonymizer (Real-time Pixelation/Censor)
[Mode 5] Holographic Vision (Cyberpunk Wireframe Effect)
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

# --- UI COLORS ---
PRIMARY_COLOR = (255, 200, 0)   # Cyan/Blue
ACCENT_COLOR = (0, 255, 0)      # Green
ALERT_COLOR = (0, 0, 255)       # Red
BG_COLOR = (15, 15, 15)
# -----------------

def draw_corner_rect(img, x1, y1, x2, y2, color, length=20, thickness=3):
    """Draws a high-tech bounding box with corner brackets."""
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def draw_sidebar(frame, fps, current_mode, model_name, is_recording):
    """Draws a unified, sci-fi transparent sidebar."""
    h, w = frame.shape[:2]
    sidebar_width = 320
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_width, 0), (w, h), BG_COLOR, -1)
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    
    # --- TOP BAR ---
    cv2.putText(frame, " THE ULTIMATE VISION SYSTEM", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRIMARY_COLOR, 2)
                
    pulse = int((np.sin(time.time() * 5) + 1) * 127) 
    cv2.circle(frame, (w - sidebar_width + 30, 20), 6, (0, pulse, 0), -1)
    cv2.putText(frame, "MULTIVERSE AI ONLINE", (w - sidebar_width + 50, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    # --- SIDEBAR CONTENT ---
    start_x = w - sidebar_width + 20
    
    cv2.putText(frame, "DASHBOARD", (start_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (start_x, 90), (w - 20, 90), PRIMARY_COLOR, 1)
    
    cv2.putText(frame, "ACTIVE ENGINE:", (start_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, model_name, (start_x, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRIMARY_COLOR, 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (start_x, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    bar_width = 200
    fill = min(int((fps / 30.0) * bar_width), bar_width)
    cv2.rectangle(frame, (start_x, 200), (start_x + bar_width, 210), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, 200), (start_x + fill, 210), ACCENT_COLOR if fps > 10 else ALERT_COLOR, -1)
    
    cv2.line(frame, (start_x, 240), (w - 20, 240), PRIMARY_COLOR, 1)
    cv2.putText(frame, "AI MODES (Press 1-5):", (start_x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Modes Menu
    modes = [
        ("1. Object Detection", 1),
        ("2. Pixel Segmentation", 2),
        ("3. Skeleton Tracking", 3),
        ("4. Privacy Censor", 4),
        ("5. Holographic Vision", 5),
        ("6. Thermal Night Vision", 6),
        ("7. Cinematic Portrait", 7)
    ]
    
    y_offset = 310
    for text, mode_num in modes:
        color = PRIMARY_COLOR if current_mode == mode_num else (100, 100, 100)
        thick = 2 if current_mode == mode_num else 1
        if current_mode == mode_num:
            cv2.rectangle(frame, (start_x, y_offset - 15), (start_x + 10, y_offset - 5), PRIMARY_COLOR, -1)
        else:
            cv2.rectangle(frame, (start_x, y_offset - 15), (start_x + 10, y_offset - 5), (100, 100, 100), 1)
            
        cv2.putText(frame, text, (start_x + 25, y_offset - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thick)
        y_offset += 35

    # Recording Indicator in Sidebar
    if is_recording:
        cv2.rectangle(frame, (start_x, h - 90), (w - 20, h - 50), (0, 0, 50), -1)
        cv2.circle(frame, (start_x + 30, h - 70), 8, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING...", (start_x + 55, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(frame, (start_x, h - 90), (w - 20, h - 50), ALERT_COLOR, 2)
        
    # --- BOTTOM BAR ---
    controls = "CONTROLS: [1-5] Modes | [S] Save Image | [R] Record Video | [Q] Quit"
    cv2.putText(frame, controls, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
    return frame

def main():
    print("=" * 70)
    print(" 🌟 INITIALIZING THE ULTIMATE VISION SYSTEM V2.0 MAX ")
    print("=" * 70)
    
    print("🔄 Loading AI Models into memory (This may take a moment)...")
    try:
        model_detect = YOLO('yolov8s-oiv7.pt')   # Mode 1, 4
        model_seg = YOLO('yolo11s-seg.pt')       # Mode 2, 5
        model_pose = YOLO('yolo11s-pose.pt')     # Mode 3
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Cannot open webcam.")
        sys.exit(1)

    print("🚀 Starting the Ultimate Vision System...")
    
    window_name = "The Ultimate AI Vision System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_time = time.time()
    fps = 0.0
    frame_count = 0
    current_mode = 1
    
    is_recording = False
    video_writer = None
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 5 == 0:
            elapsed = time.time() - fps_time
            fps = 5.0 / elapsed if elapsed > 0 else 0
            fps_time = time.time()

        # RUN INFERENCE BASED ON SELECTED MODE
        if current_mode == 1:
            active_model_name = "Supreme Detection [600 Classes]"
            results = model_detect.predict(frame, imgsz=480, conf=0.15, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = model_detect.names[int(box.cls[0])]
                draw_corner_rect(frame, x1, y1, x2, y2, PRIMARY_COLOR, length=25, thickness=2)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 10, y1), PRIMARY_COLOR, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif current_mode == 2:
            active_model_name = "Pixel Instance Segmentation"
            results = model_seg.predict(frame, imgsz=480, conf=0.3, verbose=False)
            frame = results[0].plot(boxes=False, labels=True)

        elif current_mode == 3:
            active_model_name = "Human Skeleton Pose Tracking"
            results = model_pose.predict(frame, imgsz=480, conf=0.4, verbose=False)
            frame = results[0].plot()

        elif current_mode == 4:
            active_model_name = "Privacy Anonymizer (Censor)"
            results = model_detect.predict(frame, imgsz=480, conf=0.25, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Create mosaic pixelation effect
                    small = cv2.resize(roi, (15, 15), interpolation=cv2.INTER_LINEAR)
                    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = mosaic
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_COLOR, 3)
                    cv2.putText(frame, "CENSORED", (x1 + 5, int(y1 + (y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ALERT_COLOR, 3)

        elif current_mode == 5:
            active_model_name = "Holographic Cyberpunk Vision"
            results = model_seg.predict(frame, imgsz=480, conf=0.3, verbose=False)
            if results[0].masks is not None:
                for pts in results[0].masks.xy:
                    pts = np.int32([pts])
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, pts, 255)
                    person_roi = cv2.bitwise_and(frame, frame, mask=mask)
                    edges = cv2.Canny(person_roi, 50, 150)
                    green_edges = np.zeros_like(frame)
                    green_edges[edges == 255] = [0, 255, 0]
                    frame[mask == 255] = (frame[mask == 255] * 0.2).astype(np.uint8) 
                    frame = cv2.add(frame, green_edges)
                    cv2.polylines(frame, pts, isClosed=True, color=(255, 255, 0), thickness=2)
                    
        elif current_mode == 6:
            active_model_name = "Thermal Night Vision (Heatmap)"
            results = model_detect.predict(frame, imgsz=480, conf=0.15, verbose=False)
            # Apply thermal color map to the entire frame
            thermal = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
            frame = thermal
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = model_detect.names[int(box.cls[0])]
                draw_corner_rect(frame, x1, y1, x2, y2, (255, 255, 255), length=25, thickness=2)
                cv2.putText(frame, f"{name.upper()} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif current_mode == 7:
            active_model_name = "Cinematic Portrait (Background Blur)"
            results = model_seg.predict(frame, imgsz=480, conf=0.3, classes=[0], verbose=False)
            
            if results[0].masks is not None:
                mask_all = np.zeros(frame.shape[:2], dtype=np.uint8)
                for pts in results[0].masks.xy:
                    pts = np.int32([pts])
                    cv2.fillPoly(mask_all, pts, 255)
                
                # Blur the original frame for background
                blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
                
                # Copy person from original frame
                person_fg = cv2.bitwise_and(frame, frame, mask=mask_all)
                
                # Background where mask is 0
                bg_mask = cv2.bitwise_not(mask_all)
                bg_final = cv2.bitwise_and(blurred_bg, blurred_bg, mask=bg_mask)
                
                # Combine
                frame = cv2.add(person_fg, bg_final)

        # Draw UI
        final_frame = draw_sidebar(frame, fps, current_mode, active_model_name, is_recording)

        # Record Video
        if is_recording:
            if video_writer is None:
                h, w = final_frame.shape[:2]
                filename = f"outputs/record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            video_writer.write(final_frame)
        
        cv2.imshow(window_name, final_frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif key == ord('s') or key == ord('S'):
            filename = f"outputs/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, final_frame)
        elif key == ord('r') or key == ord('R'):
            is_recording = not is_recording
            if not is_recording and video_writer is not None:
                video_writer.release()
                video_writer = None
        elif ord('1') <= key <= ord('7'):
            current_mode = key - ord('0')

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
