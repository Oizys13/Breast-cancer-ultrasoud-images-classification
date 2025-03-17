import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

# Streamlit UI Configuration
st.set_page_config(
    page_title="Breast Cancer Ultrasound Classifier",
    page_icon="🔬",
    layout="centered"
)


# Define constants
IMG_SIZE = (224, 224)
CLASSES = ["Normal", "Benign", "Malignant"]

# Load your trained model (Replace 'model.h5' with the actual model path)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/pretrained_breast_cancer_model.h5")

model = load_model()

def predict_breast_cancer(img_path, model):
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    probs = model.predict(img)[0]
    class_idx = np.argmax(probs)
    
    return {
        'class': CLASSES[class_idx],
        'confidence': float(probs[class_idx]),
        'probabilities': {c: float(p) for c, p in zip(CLASSES, probs)}
    }



# Custom CSS for styling
st.markdown(
    """
    <style>
        body { background-color: #f7f3fc; }
        .stButton > button { background-color: #957DAD; color: white; border-radius: 10px; padding: 10px; }
        .stFileUploader { border: 2px dashed #957DAD; }
        .stMarkdown { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("🩺 Breast Cancer Ultrasound Classification")
st.write("Upload an ultrasound image to classify it as Normal, Benign, or Malignant.")

# File uploader
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save temp file for prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name
    
    # Prediction Button
    if st.button("🔍 Predict Diagnosis"):
        with st.spinner("Analyzing the image..."):
            result = predict_breast_cancer(temp_path, model)
            
            # Display results
            st.subheader(f"🔬 Diagnosis: {result['class']}")
            st.write(f"Confidence: **{result['confidence']:.2%}**")
            st.write("### Probability Distribution")
            st.json(result['probabilities'])
