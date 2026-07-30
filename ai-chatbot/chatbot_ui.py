"""
🤖 Ultimate AI Chatbot (V2.0 MAX) — Streamlit Web UI
=====================================================
Powered by TinyLlama-1.1B (Large Language Model) & HuggingFace Transformers
"""
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import PyPDF2
import io

# --- Page Configuration ---
st.set_page_config(page_title="Ultimate AI Chatbot V2.0 MAX", page_icon="🤖", layout="wide")

# --- Custom CSS for a Beautiful UI ---
st.markdown("""
<style>
    .chat-bubble {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .st-emotion-cache-1c7y2kd {
        background-color: #1e1e1e !important;
    }
    h1 {
        background: -webkit-linear-gradient(#00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return pipe

# Load the pipeline
with st.spinner("Initializing AI Brain (TinyLlama-1.1B)... 🧠"):
    llm_pipeline = load_llm()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_context" not in st.session_state:
    st.session_state.doc_context = ""

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg", width=100)
    st.header("⚙️ System Control")
    mode = st.selectbox("🔮 AI Mode", ["Standard Chat", "Code Copilot 💻", "Document Q&A 📄"])
    
    st.markdown("---")
    temperature = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.7)
    
    if st.button("🗑️ Clear Memory"):
        st.session_state.messages = []
        st.session_state.doc_context = ""
        st.success("Memory cleared!")

# --- UI Header ---
st.title("🤖 Ultimate AI Chatbot (V2.0 MAX)")
st.markdown("**100% Offline | 1.1 Billion Parameters | Zero Cloud Fees**")

# Mode Specific Logic
system_prompt = "You are a helpful AI assistant."
if mode == "Standard Chat":
    system_prompt = "You are a highly intelligent, helpful, and friendly AI assistant. Answer concisely."
elif mode == "Code Copilot 💻":
    system_prompt = "You are a Senior Software Engineer. You write clean, optimized, and heavily commented code. Use markdown code blocks."
    st.info("💻 **Code Copilot Mode Active**: The AI is now optimized for programming and debugging.")
elif mode == "Document Q&A 📄":
    system_prompt = "You are a Document Analyzer. Answer questions based on the provided document context."
    st.info("📄 **Document Q&A Active**: Upload a PDF to chat with it!")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text() + " "
        st.session_state.doc_context = text[:2000] # truncate for context limit
        st.success(f"Loaded {len(st.session_state.doc_context)} characters of context!")

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
user_input = st.chat_input("Initiate neural link... (Type here)")

if user_input:
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Format the prompt
    prompt = f"<|system|>\n{system_prompt}"
    if mode == "Document Q&A 📄" and st.session_state.doc_context:
        prompt += f"\nContext: {st.session_state.doc_context}"
    prompt += "</s>\n"
    
    # Add history
    for msg in st.session_state.messages[-5:]: # Keep last 5 for context window
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    
    prompt += "<|assistant|>\n"

    # 3. Generate response
    with st.spinner("Processing in Neural Network... ⚡"):
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95
        )
        
        full_response = outputs[0]["generated_text"]
        bot_response = full_response[len(prompt):].strip()
    
    # 4. Display response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

