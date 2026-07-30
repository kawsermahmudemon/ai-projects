# 🤖 AI Projects

A collection of **7 complete, working AI/ML projects** built with Python — covering chatbots, computer vision, deep learning, generative AI, large language models, machine learning, and natural language processing.

> **No API keys required** · **CPU friendly** · **Self-contained** · **Beginner to intermediate**

---

## 📂 Projects

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 1 | [🤖 AI Chatbot](ai-chatbot/) | TF-IDF chatbot with FastAPI REST API | scikit-learn, FastAPI |
| 2 | [👁️ Computer Vision](computer-vision/) | Image classifier + webcam detection (MobileNetV2) | TensorFlow, OpenCV |
| 3 | [🧠 Deep Learning](deep-learning/) | MNIST handwritten digit recognition CNN | TensorFlow/Keras |
| 4 | [✨ Generative AI](generative-ai/) | Text generation (Markov Chain + LSTM) | TensorFlow, NumPy |
| 5 | [📚 LLM](llm/) | GPT-2 generation, sentiment, summarization | Hugging Face, PyTorch |
| 6 | [📊 Machine Learning](machine-learning/) | Iris & Titanic ML pipelines | scikit-learn, Pandas |
| 7 | [📝 NLP](natural-language-processing/) | Text preprocessing, NER, classification | NLTK, spaCy, sklearn |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/kawsermahmudemon/ai-projects.git
cd ai-projects

# Pick a project and run it
cd machine-learning
pip install -r requirements.txt
python iris_classification.py
```

Each project is **self-contained** with its own `requirements.txt` and `README.md`.

See [📖 Documentation](docs/) for detailed setup instructions.

---

## 📁 Repository Structure

```
ai-projects/
├── ai-chatbot/                     # 🤖 TF-IDF chatbot + FastAPI
│   ├── app.py                      #    Interactive terminal chatbot
│   ├── server.py                   #    REST API server
│   └── knowledge_base.json         #    Q&A knowledge base
│
├── computer-vision/                # 👁️ Image classification
│   ├── classifier.py               #    Single image classifier
│   ├── webcam_detector.py          #    Real-time webcam classifier
│   └── sample_images/              #    Sample image downloader
│
├── deep-learning/                  # 🧠 MNIST digit recognition
│   ├── model.py                    #    CNN architecture
│   ├── train.py                    #    Training script
│   ├── predict.py                  #    Prediction + drawing canvas
│   └── evaluate.py                 #    Evaluation & error analysis
│
├── generative-ai/                  # ✨ Text generation
│   ├── markov_generator.py         #    N-gram Markov chain
│   ├── lstm_generator.py           #    Character-level LSTM
│   └── data/sample_corpus.txt      #    Shakespeare training text
│
├── llm/                            # 📚 Language model tools
│   ├── text_generator.py           #    GPT-2 text generation
│   ├── sentiment_analyzer.py       #    DistilBERT sentiment
│   └── summarizer.py               #    DistilBART summarization
│
├── machine-learning/               # 📊 Classic ML pipelines
│   ├── iris_classification.py      #    Iris multi-classifier
│   ├── titanic_survival.py         #    Titanic prediction pipeline
│   └── utils.py                    #    Visualization helpers
│
├── natural-language-processing/    # 📝 NLP toolkit
│   ├── text_preprocessor.py        #    Tokenization, stemming, TF-IDF
│   ├── sentiment_analysis.py       #    VADER + Naive Bayes
│   ├── ner_extractor.py            #    Named entity recognition
│   └── text_classifier.py          #    20 Newsgroups classifier
│
├── datasets/                       # 📁 Shared datasets
│   └── titanic.csv                 #    Titanic passenger data
│
├── docs/                           # 📖 Documentation
│   └── README.md                   #    Getting started guide
│
├── assets/                         # 🎨 Shared assets
├── notebooks/                      # 📓 Jupyter notebooks
└── README.md                       # This file
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **ML/DL** | scikit-learn, TensorFlow/Keras, PyTorch |
| **NLP** | NLTK, spaCy, Hugging Face Transformers |
| **Computer Vision** | OpenCV, Pillow, MobileNetV2 |
| **Web** | FastAPI, Uvicorn |
| **Data** | Pandas, NumPy, Matplotlib, Seaborn |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
