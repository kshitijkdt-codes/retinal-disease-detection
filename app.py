from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "retinal_disease_model.h5"
IMAGE_SIZE = (224, 224)

# IMPORTANT: Keep this order identical to the class order used when training the model.
DISEASE_LABELS = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other",
]

st.set_page_config(page_title="Retinal Disease Detection", page_icon="👁️", layout="centered")


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the trained Keras model."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare an image for a ResNet50 model expecting 224x224 RGB input."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def predict(image: Image.Image, model):
    """Return class probabilities and the highest-scoring class."""
    processed = preprocess_image(image)
    predictions = np.asarray(model.predict(processed, verbose=0))

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(
            "The model output has an unexpected shape. Expected one row of class scores."
        )

    scores = predictions[0]
    if len(scores) != len(DISEASE_LABELS):
        raise ValueError(
            f"The model returned {len(scores)} scores, but the app has "
            f"{len(DISEASE_LABELS)} labels. Check the model's training class order."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid scores.")

    # This assumes the model's output is already a probability distribution (e.g. softmax).
    # If the model was trained to output logits, apply the correct conversion here.
    best_index = int(np.argmax(scores))
    results = [
        {"disease": label, "score": float(score)}
        for label, score in zip(DISEASE_LABELS, scores)
    ]
    return results, DISEASE_LABELS[best_index], float(scores[best_index])


st.title("Retinal Disease Detection")
st.write(
    "Upload a retinal fundus image to view the output of the trained image-classification model."
)

st.warning(
    "Educational project only. This model is not a medical device and must not be used "
    "to diagnose, rule out, or treat a disease. Consult a qualified eye-care professional."
)

if not MODEL_PATH.exists():
    st.error("The trained model file was not found.")
    st.markdown(
        f"Place your trained model at **`{MODEL_PATH.name}`** in the same folder as `app.py`. "
        "The model file is not included in this repository because it was not provided with "
        "the project report. See the README for details."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Check that it is a valid, compatible Keras model.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a fundus image",
    type=["jpg", "jpeg", "png"],
    help="Accepted file types: JPG, JPEG, PNG.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as a valid image. Please try another JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze image", type="primary"):
        with st.spinner("Running model prediction..."):
            try:
                results, best_label, best_score = predict(image, model)
            except Exception as exc:
                st.error("Prediction failed. Check the model output and label order.")
                st.caption(f"Technical detail: {exc}")
                st.stop()

        st.subheader("Model output")
        st.write(f"Highest-scoring class: **{best_label}**")
        st.write(f"Model score: **{best_score * 100:.2f}%**")
        st.caption(
            "The displayed score is the model's output, not a clinically validated probability "
            "of disease."
        )

        st.subheader("Scores for all classes")
        for item in sorted(results, key=lambda x: x["score"], reverse=True):
            st.write(f"{item['disease']}: {item['score'] * 100:.2f}%")
            st.progress(float(np.clip(item["score"], 0.0, 1.0)))
