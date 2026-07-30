"""
📝 NLP Titan Suite (V2.0 MAX)
==============================
Ultimate Natural Language Processing platform running entirely offline.
"""
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NLP Titan Suite V2.0 MAX", page_icon="📝", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 {
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .st-emotion-cache-1c7y2kd { background-color: #1a1a24 !important; border-radius: 10px; }
    .ner-tag { padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; margin-right: 4px; }
    .ner-PER { background-color: #FF4B4B; }
    .ner-ORG { background-color: #00C6FF; }
    .ner-LOC { background-color: #00E676; }
    .ner-MISC { background-color: #F5AF19; }
</style>
""", unsafe_allow_html=True)

st.title("📝 NLP Titan Suite (V2.0 MAX)")
st.markdown("**Powered by HuggingFace Transformers. Military-grade language analysis running 100% offline.**")

# --- Load NLP Pipelines (Cached) ---
@st.cache_resource
def load_pipelines():
    return {
        "sentiment": pipeline("sentiment-analysis"),
        "summarizer": pipeline("summarization"),
        "ner": pipeline("ner", grouped_entities=True),
        "zero_shot": pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    }

with st.spinner("Initializing Deep Language Models... 🧠"):
    pipes = load_pipelines()

# --- UI Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["😀 Sentiment Analysis", "📑 Neural Summarization", "🔍 Entity Extraction (NER)", "🎯 Zero-Shot Classification"])

with tab1:
    st.header("Emotion & Tone Detector")
    text_input = st.text_area("Input Text for Sentiment Analysis:", "The new cybernetic enhancements are absolutely breathtaking, though the price is absurdly high.", height=150)
    
    if st.button("🚀 Analyze Sentiment", use_container_width=True):
        with st.spinner("Analyzing Neural Patterns..."):
            result = pipes["sentiment"](text_input)[0]
            label = result['label']
            score = result['score'] * 100
            
            if label == "POSITIVE":
                st.success(f"**Emotion:** {label} 😊")
            else:
                st.error(f"**Emotion:** {label} 😠")
            st.progress(int(score))
            st.caption(f"AI Confidence Score: {score:.2f}%")

with tab2:
    st.header("Auto Text Summarizer")
    long_text = st.text_area("Input Long Article (Minimum 30 words):", height=200)
    
    if st.button("⚡ Condense Data", use_container_width=True):
        if len(long_text.split()) < 30:
            st.warning("Insufficient data payload. Please provide at least 30 words.")
        else:
            with st.spinner("Compressing text dimensions..."):
                summary = pipes["summarizer"](long_text, max_length=130, min_length=30, do_sample=False)
                st.info("✅ Summary Extracted:")
                st.write(summary[0]['summary_text'])

with tab3:
    st.header("Named Entity Recognition (NER)")
    st.markdown("Instantly identify People, Organizations, and Locations from raw text.")
    ner_text = st.text_area("Input Text for Entity Extraction:", "Elon Musk founded SpaceX in Hawthorne, California.", height=150)
    
    if st.button("🕵️ Extract Entities", use_container_width=True):
        with st.spinner("Scanning for entities..."):
            entities = pipes["ner"](ner_text)
            
            if not entities:
                st.warning("No named entities detected.")
            else:
                st.markdown("### Extracted Entities")
                for ent in entities:
                    tag_class = f"ner-{ent['entity_group']}"
                    st.markdown(f"<span class='ner-tag {tag_class}'>{ent['entity_group']}</span> {ent['word']} (Confidence: {ent['score']*100:.1f}%)", unsafe_allow_html=True)
                    st.write("") # spacing

with tab4:
    st.header("Zero-Shot Classification")
    st.markdown("Classify text into *any* categories you define, without prior training!")
    
    col_t, col_l = st.columns(2)
    with col_t:
        zs_text = st.text_area("Text to Classify:", "The stock market plunged today after tech companies reported lower earnings.", height=150)
    with col_l:
        labels_str = st.text_input("Categories (comma-separated):", "finance, sports, technology, politics")
        
    if st.button("🎯 Classify Document", use_container_width=True):
        labels_list = [l.strip() for l in labels_str.split(",") if l.strip()]
        if not labels_list:
            st.error("Please provide at least one category.")
        else:
            with st.spinner("Running Zero-Shot Inferencing..."):
                result = pipes["zero_shot"](zs_text, candidate_labels=labels_list)
                
                st.markdown("### Classification Results")
                for label, score in zip(result['labels'], result['scores']):
                    st.write(f"**{label.title()}**")
                    st.progress(int(score * 100))
                    st.caption(f"Probability: {score*100:.2f}%")
