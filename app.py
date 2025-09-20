import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------------
# Load the trained model
# ------------------------------
MODEL_PATH = "image_classification_model.keras"
model = load_model(MODEL_PATH)

# Load class labels (must match your training generator)
class_labels = ['fresh', 'ripe', 'rotten', 'unripe']  # adjust order if needed

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fruit Quality Classifier", layout="centered")
st.title("🍎🥭 Fruit Classification App")
st.write("Upload an image of a fruit and the model will classify it as **Fresh, Rotten, Ripe, Unripe, or Disease**.")

# File uploader
uploaded_file = st.file_uploader("Upload a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Show results
    st.markdown(f"### ✅ Predicted Class: **{class_labels[class_id]}**")
    st.markdown(f"### 🔥 Confidence: **{confidence*100:.2f}%**")

    # Show probabilities as a bar chart
    st.bar_chart(predictions[0])
