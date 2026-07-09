import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🎗️",
    layout="centered"
)


# --------------------------------------------------
# LOAD YOUR TRAINED MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(
    "best_model_class_weights.keras"
)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "benign",
    "malignant",
    "normal"
]


# --------------------------------------------------
# PINK AND WHITE GRADIENT DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main pink and white gradient background */

.stApp {
    background: linear-gradient(
        135deg,
        #ffb6c1 0%,
        #ffc0cb 30%,
        #ffe4e9 65%,
        #ffffff 100%
    );
    background-attachment: fixed;
}


/* Main title */

h1 {
    color: #ad1457 !important;
    text-align: center;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
}


/* Normal text */

p {
    color: #6d214f;
    font-size: 17px;
}


/* File uploader */

[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #f48fb1;
    box-shadow: 0px 8px 20px rgba(194, 24, 91, 0.15);
}


/* File uploader text */

[data-testid="stFileUploader"] p {
    color: #880e4f !important;
}


/* Subheadings */

h2, h3 {
    color: #ad1457 !important;
}


/* Uploaded image */

[data-testid="stImage"] img {
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(136, 14, 79, 0.20);
}


/* Result boxes */

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FRONTEND TITLE
# --------------------------------------------------

st.title(" Breast Cancer Detection System")

st.write(
    "Upload a breast cancer image to predict whether it is "
    "**Benign, Malignant, or Normal**."
)


# --------------------------------------------------
# IMAGE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Breast Cancer Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image and convert it to RGB
    image = Image.open(uploaded_file).convert("RGB")


    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Breast Cancer Image",
        use_container_width=True
    )


    # Resize image to the size used during model training
    img = image.resize((224, 224))


    # Convert image into NumPy array
    img_array = np.array(img)


    # Normalize image pixel values from 0-255 to 0-1
    img_array = img_array / 255.0


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Make prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )


    # Get predicted class index
    predicted_index = np.argmax(prediction)


    # Convert index into class name
    predicted_class = class_names[predicted_index]


    # Calculate confidence percentage
    confidence = np.max(prediction) * 100


    # --------------------------------------------------
    # DISPLAY PREDICTION RESULT
    # --------------------------------------------------

    st.subheader("🔬 Prediction Result")


    st.success(
        f"Predicted Class: {predicted_class.upper()}"
    )


    st.info(
        f"Confidence Score: {confidence:.2f}%"
    )


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.warning(
    "⚠️ Disclaimer: This application is developed for academic "
    "project demonstration purposes only. It should not be used "
    "as a substitute for professional medical diagnosis."
)
