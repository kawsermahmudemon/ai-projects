# 📖 Documentation

## Getting Started

### Prerequisites

- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **pip** — Comes with Python (used for installing dependencies)
- **Git** — [Download Git](https://git-scm.com/)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kawsermahmudemon/ai-projects.git
cd ai-projects

# Navigate to any project
cd machine-learning

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python iris_classification.py
```

### Per-Project Setup

Each project has its own `requirements.txt`. Install dependencies for each project separately:

```bash
# Example: Set up the deep learning project
cd deep-learning
pip install -r requirements.txt
python train.py
```

### Common Dependencies

| Library | Used In | Purpose |
|---------|---------|---------|
| scikit-learn | ML, NLP, Chatbot | ML algorithms, metrics, preprocessing |
| tensorflow | CV, DL, GenAI | Neural networks, pre-trained models |
| transformers | LLM | Hugging Face model inference |
| matplotlib | All | Visualization |
| numpy | All | Numerical computing |
| pandas | ML, NLP | Data manipulation |
| nltk | NLP | Text processing |
| spacy | NLP | Named entity recognition |
| opencv-python | CV, DL | Image/video processing |
| fastapi | Chatbot | REST API |

### Notes

- **First run downloads** — Some projects download pre-trained models on first run (MobileNetV2 ~14MB, GPT-2 ~500MB, DistilBERT ~260MB, DistilBART ~1.2GB)
- **CPU only** — All projects run on CPU. GPU is not required
- **No API keys** — No paid services or API keys needed
- **Python 3.10+** — Type hints use modern syntax (`list[str]` instead of `List[str]`)
