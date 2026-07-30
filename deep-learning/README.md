# 🧠 Ultimate Deep Learning Studio (V2.0 MAX)

Welcome to the **Ultimate Deep Learning Studio V2.0 MAX**, a state-of-the-art vision analysis platform.

This project uses Deep Convolutional Neural Networks via PyTorch to classify any uploaded image into one of 1000 categories, offering multiple architectures and advanced feature map visualizations.

---

## ✨ Jaw-Dropping Features

- **🌐 Multi-Model Architecture:** Instantly switch between MobileNet V2 (Fast), ResNet-50 (Balanced), and DenseNet-121 (Accurate) without restarting!
- **🔬 Neural Edge Activation Maps:** Peek inside the "mind" of the AI by visualizing the low-level edge features it extracts from your image using pseudo-thermal activation colormaps.
- **📊 Interactive Confidence Bars:** View the top 5 predictions with beautiful progress bars showing exact confidence percentages.
- **⚡ Offline Inference:** Runs entirely locally on your CPU/GPU. No internet required after initial weights download.
- **🎨 Beautiful Cyber UI:** A stunning dark-mode Streamlit interface with gradient headers.

---

## 🚀 How to Run

1. Open your terminal or command prompt.
2. Navigate to the project directory:
   ```cmd
   cd c:\Github\ai-projects\deep-learning
   ```
3. Activate the virtual environment:
   ```cmd
   .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
5. Start the Web Interface:
   ```cmd
   streamlit run app.py
   ```

---

## 🛠️ Technologies Used
* **Python 3**
* **PyTorch & Torchvision** (Deep Learning Backend)
* **OpenCV** (For Neural Edge Activation visualizations)
* **Streamlit** (UI Framework)
