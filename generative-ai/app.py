"""
🎨 Ultimate Generative AI Studio (V2.0 MAX)
=============================================
Instant Text-to-Image Generation with Seed control and Cyber UI.
"""
import streamlit as st
import urllib.parse
import requests
from PIL import Image
from io import BytesIO
import random
import time

st.set_page_config(page_title="Ultimate Image Forge V2.0 MAX", page_icon="🎨", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 {
        background: -webkit-linear-gradient(#f12711, #f5af19);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f12711 0%, #f5af19 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(245, 175, 25, 0.6);
    }
    .st-emotion-cache-1c7y2kd { background-color: #1a1a24 !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Ultimate Image Forge (V2.0 MAX)")
st.markdown("**Harness the power of State-of-the-Art Diffusion Models. No API Keys. Zero Limits.**")

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Forge Parameters")
    
    prompt = st.text_area("🌟 Positive Prompt", "A futuristic cyberpunk city at night, flying cars, neon lights, highly detailed, photorealistic", height=100)
    negative_prompt = st.text_input("🚫 Negative Prompt (What to avoid)", "blurry, distorted, low quality, deformed, ugly")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        style = st.selectbox("🎨 Art Style", ["Photorealistic", "Anime / Manga", "Cyberpunk", "Cinematic", "3D Render", "Pixel Art"])
    with col_b:
        aspect_ratio = st.selectbox("📐 Aspect Ratio", ["Square (1024x1024)", "Portrait (768x1024)", "Landscape (1024x768)", "Ultrawide (1920x1080)"])
        
    use_seed = st.checkbox("🎲 Use Custom Seed")
    seed = st.number_input("Seed Value", value=42) if use_seed else random.randint(1, 9999999)
    
    generate_btn = st.button("🚀 IGNITE THE FORGE (Generate)")

with col2:
    st.header("🖼️ Masterpiece Canvas")
    
    # Aspect Ratio Logic
    width, height = 1024, 1024
    if "Portrait" in aspect_ratio: width, height = 768, 1024
    if "Landscape" in aspect_ratio: width, height = 1024, 768
    if "Ultrawide" in aspect_ratio: width, height = 1920, 1080

    if generate_btn and prompt:
        # UX Polish: Fake progress bar for anticipation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Connecting to Neural Synthesis Engine...")
        for i in range(1, 60, 2):
            time.sleep(0.01)
            progress_bar.progress(i)
            
        status_text.text("Diffusing Latent Space...")
        
        # Build prompt
        style_modifiers = {
            "Photorealistic": "photorealistic, 8k resolution, raw photo, highly detailed, sharp focus, masterpiece",
            "Anime / Manga": "anime artwork, studio ghibli, makoto shinkai, colorful, beautiful shading, anime style",
            "Cyberpunk": "cyberpunk 2077 style, neon lighting, dark, futuristic, dystopian, blade runner",
            "Cinematic": "cinematic lighting, dramatic, movie scene, depth of field, volumetric lighting",
            "3D Render": "unreal engine 5, octane render, 3d modeling, smooth, ray tracing, 4k",
            "Pixel Art": "16-bit pixel art, retro gaming style, high contrast, clean pixels"
        }
        
        enhanced_prompt = f"{prompt}, {style_modifiers[style]} | negative: {negative_prompt}"
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={seed}&nologo=true"
        
        try:
            response = requests.get(image_url)
            
            for i in range(60, 101, 5):
                time.sleep(0.02)
                progress_bar.progress(i)
                
            status_text.text("Finalizing Pixels...")
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=f'Prompt: "{prompt}" | Seed: {seed}', use_container_width=True)
                
                status_text.empty()
                progress_bar.empty()
                
                st.download_button(
                    label="💾 Download Ultra-HD Image",
                    data=response.content,
                    file_name=f"forge_img_{seed}.png",
                    mime="image/png"
                )
            else:
                st.error("Neural Engine Overloaded. Please try again.")
        except Exception as e:
            st.error(f"Critical System Failure: {e}")
            
    elif not generate_btn:
        st.info("👈 Configure your Neural Parameters and click 'IGNITE THE FORGE'!")
        # Sample Gallery
        st.markdown("### 🔥 Recent Community Masterpieces")
        g1, g2, g3 = st.columns(3)
        g1.image("https://image.pollinations.ai/prompt/cyberpunk%20city%20night%20flying%20cars?width=512&height=512&nologo=true", caption="Cyberpunk City")
        g2.image("https://image.pollinations.ai/prompt/beautiful%20anime%20girl%20glowing%20eyes?width=512&height=512&nologo=true", caption="Anime Aesthetic")
        g3.image("https://image.pollinations.ai/prompt/unreal%20engine%205%20fantasy%20castle?width=512&height=512&nologo=true", caption="Cinematic Fantasy")
