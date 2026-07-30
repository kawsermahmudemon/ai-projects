"""
🧠 Ultimate LLM Toolkit (V2.0 MAX)
=====================================
Unified Large Language Model dashboard running locally.
"""
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Ultimate LLM Toolkit V2.0 MAX", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 {
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .st-emotion-cache-1c7y2kd { background-color: #1a1a24 !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Ultimate LLM Toolkit (V2.0 MAX)")
st.markdown("**Your all-in-one local LLM inference dashboard. No internet required.**")

@st.cache_resource
def load_models():
    return {
        "generator": pipeline('text-generation', model='gpt2'),
        "sentiment": pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english'),
        "summarizer": pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    }

with st.spinner("Loading Transformer Models into Memory..."):
    models = load_models()

tab1, tab2, tab3 = st.tabs(["✍️ Neural Text Generation", "🎭 Sentiment Analysis", "📑 Document Summarization"])

with tab1:
    st.header("GPT-2 Text Generation")
    prompt = st.text_area("Enter a prompt:", "In the year 2050, Artificial Intelligence...")
    
    col_t, col_l = st.columns(2)
    with col_t:
        temp = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.8)
    with col_l:
        length = st.slider("Max Length", 50, 500, 150)
        
    if st.button("Generate Text 🚀"):
        with st.spinner("Generating..."):
            res = models["generator"](prompt, max_length=length, temperature=temp, do_sample=True, num_return_sequences=1)
            st.success("✅ Generation Complete!")
            st.write(res[0]['generated_text'])

with tab2:
    st.header("DistilBERT Sentiment Analysis")
    sentiment_text = st.text_area("Enter text to analyze:", "This unified toolkit is absolutely incredible!")
    
    if st.button("Analyze Sentiment 🔍"):
        with st.spinner("Analyzing..."):
            res = models["sentiment"](sentiment_text)[0]
            if res['label'] == 'POSITIVE':
                st.success(f"**Emotion:** POSITIVE 😊 (Confidence: {res['score']*100:.1f}%)")
            else:
                st.error(f"**Emotion:** NEGATIVE 😠 (Confidence: {res['score']*100:.1f}%)")

with tab3:
    st.header("DistilBART Summarization")
    summary_text = st.text_area("Enter long document:", height=200)
    
    if st.button("Summarize Document 📑"):
        if len(summary_text.split()) < 30:
            st.warning("Please enter at least 30 words.")
        else:
            with st.spinner("Summarizing..."):
                res = models["summarizer"](summary_text, max_length=130, min_length=30, do_sample=False)
                st.info("✅ Summary:")
                st.write(res[0]['summary_text'])
