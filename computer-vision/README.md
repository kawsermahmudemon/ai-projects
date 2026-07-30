# 🌟 Ultimate AI Vision System (V2.0 MAX)

Welcome to the **Ultimate AI Vision System**, a state-of-the-art real-time computer vision application built with Python, OpenCV, and Ultralytics YOLOv11/YOLOv8! 

This project transforms your standard webcam into an incredibly powerful AI camera capable of running 7 jaw-dropping vision models in real-time.

---

## ✨ Features & AI Modes

Press the numbers **`1` to `7`** on your keyboard to instantly switch between these incredible AI modes:

### 1️⃣ Supreme Detection [600 Classes]
Uses a zero-shot model (YOLO-World) to detect over 600 everyday objects with extreme accuracy. It draws bounding boxes and confidence scores in real-time.

### 2️⃣ Pixel-Perfect Segmentation
Uses the YOLO11 Segmentation model to not just draw a box, but to perfectly trace the exact pixel outline of people and objects, highlighting them with a colorful translucent mask.

### 3️⃣ Human Skeleton Tracking
Uses the YOLO11 Pose Estimation model to detect human bodies and map out their skeleton (joints, arms, legs, face) in real-time.

### 4️⃣ Privacy Anonymizer (Censor Mode)
Automatically detects people in the frame and heavily pixelates/censors them, drawing a red "CENSORED" warning. Perfect for privacy-focused video feeds.

### 5️⃣ Holographic Cyberpunk Vision
Extracts the human body from the background, darkens it, and wraps it in a glowing neon-green digital wireframe (Canny edges), creating a Sci-Fi Matrix/Hologram effect.

### 6️⃣ Thermal Night Vision (Heatmap)
Converts your camera into a Predator-style thermal imaging camera using OpenCV's Inferno colormap, while still running object detection in the dark.

### 7️⃣ Cinematic Portrait (Background Blur)
Mimics a high-end DSLR camera or iPhone Portrait Mode. It uses the segmentation mask to perfectly isolate you and heavily blurs the entire background!

---

## 🎮 Controls

While the camera is running, you have full control using your keyboard:

- **`1` - `7`**: Switch between the AI modes instantly.
- **`R`**: Start or Stop **Video Recording**. The video will be saved as an `.mp4` file in the `outputs/` folder.
- **`S`**: Instantly save a high-quality **Screenshot** (`.jpg`) to the `outputs/` folder.
- **`Q`**: Quit the application safely.

---

## 🚀 How to Run

1. Open your terminal or command prompt.
2. Navigate to the project directory:
   ```cmd
   cd c:\Github\ai-projects\computer-vision
   ```
3. Activate the virtual environment:
   ```cmd
   .\venv\Scripts\activate
   ```
4. Run the Ultimate Vision System:
   ```cmd
   python ultimate_vision_system.py
   ```

*(Note: The first time you run it, it might take a few moments to automatically download the required AI models).*

---

## 🛠️ Technologies Used
* **Python 3**
* **OpenCV** (Image processing, UI, and rendering)
* **Ultralytics (YOLO11 & YOLOv8-World)** (Deep Learning models)
* **NumPy** (Matrix mathematics and pixel manipulation)

Enjoy exploring the cutting-edge of AI computer vision! 🚀
