import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")
st.title("🔍 AI vs Real Image Detector")
st.write("Upload an image to check whether it is AI-generated or Real.")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/efficientnet_clean.keras",
        compile=False
    )
    return model

model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
IMG_SIZE = 224

def preprocess_image(image):
    """
    CRITICAL: EfficientNetB0 has built-in preprocessing layers.
    Pass RAW pixels (0-255), NOT normalized or preprocessed.
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")  # Raw pixels 0-255
    # NO preprocess_input! EfficientNet does this internally
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        st.subheader("Prediction Result:")
        
        # Check class mapping from your training
        # If your model outputs: >0.5 = class 1, <0.5 = class 0
        # And if class_names are ['FAKE', 'REAL'] (alphabetical)
        # Then: >0.5 = REAL, <0.5 = FAKE
        
        if prediction >= 0.5:
            confidence = prediction * 100
            st.success(f"✅ **REAL Image**")
            st.write(f"Confidence: **{confidence:.1f}%**")
            st.progress(prediction)
        else:
            confidence = (1 - prediction) * 100
            st.error(f"🤖 **FAKE / AI-Generated Image**")
            st.write(f"Confidence: **{confidence:.1f}%**")
            st.progress(1 - prediction)
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write(f"Raw sigmoid output: {prediction:.6f}")
            st.write(f"Interpretation: >0.5 = REAL, <0.5 = FAKE")

