import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 64

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for polished dark/light UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .real-card {
        background-color: rgba(40, 167, 69, 0.15);
        border: 2px solid #28a745;
        color: #28a745;
    }
    .fake-card {
        background-color: rgba(220, 53, 69, 0.15);
        border: 2px solid #dc3545;
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_deepfake_model():
    """Load and cache the trained Deepfake classification Keras model."""
    try:
        model = tf.keras.models.load_model('deepfake_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model file 'deepfake_model.keras': {e}")
        return None

# App Header
st.markdown('<div class="main-title">🕵️ Deepfake vs Real Face Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered CNN Model for Image Authenticity Verification</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI model..."):
    model = load_deepfake_model()

if model is None:
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("About Model")
    st.write("**Architecture:** Deep CNN with Batch Normalization")
    st.write("**Input Size:** 64x64 RGB")
    st.write("**Classes:** Real Image vs Deepfake Image")
    st.divider()
    st.caption("Deepfake Detection System v1.0")

# Upload section
st.write("### Upload Image")
uploaded_file = st.file_uploader(
    "Choose a face image to analyze (JPG, JPEG, PNG)...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Analysis Result")
        with st.spinner("Analyzing image features..."):
            # Image preprocessing matching notebook pipeline
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized, dtype='float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prob = model.predict(img_array, verbose=0)[0][0]
            label = "Real Image" if prob > 0.5 else "Deepfake / Synthetic"
            confidence = prob if prob > 0.5 else 1.0 - prob
            confidence_pct = confidence * 100

        # Display result badge
        if prob > 0.5:
            st.success(f"🟢 **Verdict:** {label}")
        else:
            st.error(f"🔴 **Verdict:** {label}")

        st.metric(label="Model Confidence Score", value=f"{confidence_pct:.2f}%")
        st.progress(float(confidence))

        with st.expander("Technical Prediction Details"):
            st.write(f"- **Raw Probability Score:** `{prob:.4f}`")
            st.write(f"- **Threshold:** `0.50` (Real > 0.50, Fake ≤ 0.50)")
            st.write(f"- **Input Tensor Shape:** `{img_array.shape}`")
else:
    st.info("👆 Please upload an image above to start the deepfake verification analysis.")
