import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle

# Constants
IMG_SIZE = 128  # Must match training

# Load model and label encoder only once
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("dog_breed_classifier.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()

# Preprocess uploaded image
def preprocess_image(image_bytes):
    # Decode image (ensure RGB for model)
    img_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:  # fallback for certain file types
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a dog photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    file_bytes = uploaded_file.read()
    img_np = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=300)
    else:
        st.error("Could not decode image. Please upload a valid image.")
    
    # Preprocess and predict
    img_proc = preprocess_image(file_bytes)
    if img_proc is not None:
        with st.spinner('Predicting breed...'):
            pred_proba = model.predict(img_proc)[0]
            pred_idx = np.argmax(pred_proba)
            breed = le.inverse_transform([pred_idx])[0]
            confidence = pred_proba[pred_idx]
        st.markdown(f"### 🐾 Predicted breed: **{breed}** (confidence: {confidence:.2%})")
    else:
        st.warning("Failed to preprocess the uploaded image.")

st.markdown("---")
st.caption("Upload a dog image above to identify its breed using your custom-trained model.")