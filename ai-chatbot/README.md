# 🤖 Ultimate AI Chatbot (V2.0 MAX)

Welcome to the **Ultimate AI Chatbot V2.0 MAX**, an incredibly powerful, fully offline, and 100% free Large Language Model (LLM) application. 

This project completely replaces standard TF-IDF and basic machine learning with **TinyLlama-1.1B**—a state-of-the-art 1.1 Billion parameter neural network running entirely on your local machine's CPU!

---

## ✨ Jaw-Dropping Features (3 AI Modes)

- **💬 1. Standard Chat Mode:** The AI remembers your entire chat history and context during the session. It can answer questions, write stories, and chat fluently.
- **💻 2. Code Copilot Mode:** Optimized for programming! The AI acts as a Senior Software Engineer, writing clean, optimized, and heavily commented code with beautiful syntax highlighting.
- **📄 3. Document Q&A (RAG) Mode:** Upload a PDF document directly into the chat, and the AI will analyze its text and answer questions specifically based on your document's context!
- **🔒 100% Private & Offline:** No internet connection required after the initial model download. No API keys, no monthly fees, and no data leaves your computer.
- **🎨 Beautiful Cyber Web UI:** Built with Streamlit, providing a ChatGPT-like interface with a sleek dark mode layout and gradient headers.

---

## 🚀 How to Run

1. Open your terminal or command prompt.
2. Navigate to the project directory:
   ```cmd
   cd c:\Github\ai-projects\ai-chatbot
   ```
3. Activate the virtual environment (if using one):
   ```cmd
   .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
5. Start the Streamlit Web Application:
   ```cmd
   streamlit run chatbot_ui.py
   ```

*(Note: The very first time you run this, it will automatically download the TinyLlama model weights (~2.2 GB) from HuggingFace. Please be patient while it downloads!)*

---

## 🛠️ Technologies Used
* **Python 3**
* **Streamlit** (For the beautiful Chat UI)
* **HuggingFace Transformers** (For LLM pipeline and Tokenization)
* **PyTorch** (Deep learning backend)
* **PyPDF2** (For Document Parsing & RAG)
* **TinyLlama-1.1B-Chat** (The core AI brain)
