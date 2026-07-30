# 🧠 Ultimate LLM Toolkit (V2.0 MAX)

Welcome to the **Ultimate LLM Toolkit V2.0 MAX**! This is your all-in-one local LLM inference dashboard. No internet required, no API keys, pure neural power.

## ✨ Jaw-Dropping Features
- **✍️ Neural Text Generation:** Generate creative text continuations using GPT-2 with adjustable creativity (temperature) and length.
- **🎭 Sentiment Analysis:** Analyze the tone of any text using DistilBERT with a slick UI and confidence scoring.
- **📑 Document Summarization:** Condense long articles into short, precise paragraphs instantly using DistilBART.
- **⚡ 100% Offline & Private:** Runs entirely locally on your CPU/GPU.
- **🌌 Cyber UI:** Breathtaking dark-mode Streamlit UI with unified tabs.

## 🚀 How to Run

```cmd
cd c:\Github\ai-projects\llm
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** The first run will automatically download the transformer models (~2GB total) from HuggingFace to your local cache. Subsequent runs will be instant!
