"""
Streamlit app - Brain Tumor MRI Classifier (VGG16 transfer learning, Optuna-tuned)
 
Run locally with:
    streamlit run app.py
 
Looks for a trained model file in the same folder as this script (or in a
"models/" subfolder), trying a few known filenames in order — see
MODEL_CANDIDATES below. If you rename or move your model file, add its name
there too.
"""
 
import os
 
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
 
# ----------------------------------------------------------------------------
# Configuration - must match what the model was trained with
# ----------------------------------------------------------------------------
IMG_SIZE = (150, 150)
# Alphabetical order - matches CLASS_NAMES = sorted(os.listdir(TRAIN_DIR)) in the notebook
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
 
CLASS_DESCRIPTIONS = {
    "glioma": "A tumor arising from glial (supporting) cells in the brain or spine.",
    "meningioma": "A tumor arising from the meninges, the membranes around the brain and spinal cord.",
    "notumor": "No visible tumor detected in the scan.",
    "pituitary": "A tumor arising in the pituitary gland at the base of the brain.",
}
 
# Tried in this order, in the current folder AND inside a "models/" subfolder.
# Keeping several names here means the app still works no matter which of your
# saved model files you end up keeping.
MODEL_CANDIDATES = [
    "best_model_vgg16_transfer.keras",  # name the training notebook saves by default
    "best_brain_tumor_model.keras",
    "vgg16_brain_mri.keras",
    "vgg16_brain_mri.h5",
]
 
 
def resolve_model_path() -> str:
    """Return the first model file that actually exists on disk."""
    search_paths = []
    for name in MODEL_CANDIDATES:
        search_paths.append(name)
        search_paths.append(os.path.join("models", name))
 
    for path in search_paths:
        if os.path.exists(path):
            return path
 
    # Nothing found - return the first candidate so the error message below
    # at least tells the user what filename it expected.
    return MODEL_CANDIDATES[0]
 
 
MODEL_PATH = resolve_model_path()
 
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
 
 
# ----------------------------------------------------------------------------
# Model loading - cached so it only loads once per session
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)
 
 
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + rescale an uploaded image to match the training pipeline exactly."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # add batch dimension
 
 
# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI scan and the model will predict whether it shows a "
    "**glioma**, **meningioma**, or **pituitary** tumor, or **no tumor**."
)
 
st.warning(
    "⚠️ This is a student capstone project, not a medical device. Its predictions "
    "are not a diagnosis and should never be used to make real clinical decisions. "
    "Always consult a qualified radiologist or physician."
)
 
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
    st.sidebar.caption(f"Model file in use:\n`{MODEL_PATH}`")
except Exception as e:
    model_loaded = False
    st.error(
        f"Could not load a model. Looked for: {', '.join(MODEL_CANDIDATES)} "
        f"(in this folder and in a 'models/' subfolder). Make sure one of these "
        f"files is present — it's produced by running the training notebook end to end.\n\n"
        f"Details: {e}"
    )
 
uploaded_file = st.file_uploader("Choose an MRI image (jpg / png)", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded scan", use_container_width=True)
 
    if model_loaded:
        with st.spinner("Analyzing scan..."):
            batch = preprocess_image(pil_image)
            probs = model.predict(batch, verbose=0)[0]
 
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
 
        with col2:
            st.subheader("Prediction")
            st.metric(label="Predicted class", value=pred_class, delta=f"{confidence:.1%} confidence")
            st.caption(CLASS_DESCRIPTIONS[pred_class])
 
        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        }).sort_values("Probability", ascending=False).set_index("Class")
        st.bar_chart(prob_df)
 
        with st.expander("Raw probability values"):
            st.dataframe(
                prob_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True
            )
else:
    st.info("Upload an MRI image above to get a prediction.")
 
st.divider()
st.caption(
    "Model: VGG16 (transfer learning, ImageNet weights), classifier head tuned with "
    "Optuna, fine-tuned on the top VGG16 layers. Trained on a 4-class brain MRI dataset "
    "(glioma, meningioma, notumor, pituitary)."
)
