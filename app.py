import streamlit as st
import os
import keras
import numpy as np
from PIL import Image
from keras.models import load_model
from utils import preprocess_image, classify_image

# Path to the model file 
model_path = os.path.join("models", "MRI-brain-tumor-detection-model.h5")

# Loading the pre-trained model from the notebook 
model = load_model(model_path)

# Background Image
def set_bg_hack_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with the image URL
set_bg_hack_url("https://cdn.pixabay.com/photo/2020/06/19/22/33/wormhole-5319067_960_720.jpg")



st.title("MRI Brain Tumor Classifier")

uploaded_image = st.file_uploader("Hello! Upload an 'JPG' MRI brain image: ", type=["jpg"])

if uploaded_image is not None:

    image = preprocess_image(uploaded_image)
    prediction, confidence = classify_image(model, image)

    st.image(image, caption=f"Prediction: {prediction} ({confidence:.2%})")

