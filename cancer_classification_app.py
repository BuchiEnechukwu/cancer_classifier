
import os 
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow import keras

# Debugging
#st.caption(f"TF: {tf.__version__} | NumPy: {np.__version__}")

# Page config
st.set_page_config(page_title="Welcome to OncoData", layout="wide")

# inject CSS
def inject_css():
    st.markdown("""
    <style>
      .main .block-container {max-width: 1100px; padding-top: 1.5rem; padding-bottom: 2.25rem;}

      /* Sidebar */
      section[data-testid="stSidebar"] {
          background: #F6FBFA;
          border-right: 1px solid #E2F1EE;
          width: 250px !important;    /* force sidebar width */
      }
      section[data-testid="stSidebar"] > div:first-child {
          max-width: 250px !important;  /* keep content compact */
      }

      .sidebar-card {background:#ECF8F6; border:1px solid #D4EEE9; border-radius:14px; padding:12px; margin-bottom:12px;}
      .nav-btn > button {width:100%; border-radius:12px; font-weight:600;}

      /* Cards in main area */
      .card {background:#FFFFFF; border:1px solid #E6EEF0; border-radius:18px; padding:1.2rem;
             box-shadow:0 6px 18px rgba(16,24,40,.06); margin-bottom:1rem;}

      h1,h2,h3 {letter-spacing:.2px}
    </style>
    """, unsafe_allow_html=True)
inject_css()


# App Config
IMG_SIZE = (128, 128)
MODEL_PATH = "runs/cancer_classifier_model.keras"
LABELS_PATH = "runs/label_encoder.pkl"

# Optional: human-readable descriptions (keys must match your class names)
LABEL_DESCRIPTIONS = {
    "all_benign": "Benign blood cells (non-cancerous)",
    "all_early": "Early-stage acute lymphoblastic leukemia",
    "all_pre": "Pre-B cell subtype of leukemia",
    "all_pro": "Pro-B cell subtype of leukemia",
    "brain_glioma": "Glioma (tumor from glial cells)",
    "brain_menin": "Meningioma (tumor from meninges)",
    "brain_tumor": "General brain tumor",
    "breast_benign": "Benign breast tissue",
    "breast_malignant": "Malignant breast tissue (cancerous)",
    "cervix_dyk": "Dyskeratotic cells (abnormal keratinization)",
    "cervix_koc": "Koilocytotic cells (HPV-related changes)",
    "cervix_mep": "Metaplastic epithelial cells",
    "cervix_pab": "Parabasal cells (immature squamous cells)",
    "cervix_sfi": "Superficial squamous cells (normal)",
    "colon_aca": "Colon adenocarcinoma (colon cancer)",
    "colon_bnt": "Benign colon tissue",
    "kidney_normal": "Healthy kidney tissue",
    "kidney_tumor": "Kidney tumor (cancerous)",
    "lung_aca": "Lung adenocarcinoma",
    "lung_bnt": "Benign lung tissue",
    "lung_scc": "Lung squamous cell carcinoma",
    "lymph_cll": "Chronic lymphocytic leukemia",
    "lymph_fl": "Follicular lymphoma",
    "lymph_mcl": "Mantle cell lymphoma",
    "oral_normal": "Healthy oral tissue",
    "oral_scc": "Oral Squamous Cell Carcinoma"
}

# Human-friendly display names for your classes
DISPLAY_OVERRIDES = {
    "all_benign": "ALL Benign",
    "all_early": "ALL Early",
    "all_pre": "ALL Pre-B",
    "all_pro": "ALL Pro-B",
    "brain_glioma": "Brain Glioma",
    "brain_menin": "Brain Meningioma",
    "brain_tumor": "Brain Tumor",
    "breast_benign": "Breast Benign",
    "breast_malignant": "Breast Malignant",
    "cervix_dyk": "Cervix Dyskeratotic",
    "cervix_koc": "Cervix Koilocytotic",
    "cervix_mep": "Cervix Metaplastic",
    "cervix_pab": "Cervix Parabasal",
    "cervix_sfi": "Cervix Superficial",
    "colon_aca": "Colon Adenocarcinoma",
    "colon_bnt": "Colon Benign Tissue",
    "kidney_normal": "Kidney Normal",
    "kidney_tumor": "Kidney Tumor",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_bnt": "Lung Benign Tissue",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "lymph_cll": "Lymph CLL",
    "lymph_fl": "Lymph Follicular Lymphoma",
    "lymph_mcl": "Lymph Mantle Cell Lymphoma",
    "oral_normal": "Oral Normal",
    "oral_scc": "Oral Squamous Cell Carcinoma",
}

def pretty_label(name: str) -> str:
    # Use overrides where provided; otherwise convert snake_case -> Title Case
    return DISPLAY_OVERRIDES.get(name, name.replace("_", " ").title())

#  Cached loaders 
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_classes(path: str):
    label_map = joblib.load(path)
    classes = [None]*len(label_map)
    for name, idx in label_map.items():
        classes[idx] = name
    return classes

#  Sidebar 
with st.sidebar:
    if Path("assets/logo.png").exists():
        st.image("assets/logo.png", width=120)

    st.markdown("<div class='sidebar-card'><b>🔐 Login</b></div>", unsafe_allow_html=True)
    access = st.text_input("Enter access key", type="password")
    st.caption("Empowering diagnostics through intelligent technology")

    st.markdown("<div class='sidebar-card'><b>🧭 OncoData Navigation</b></div>", unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    st.button("Home", key="nav_home", on_click=lambda: st.session_state.update(page="Home"))
    st.button("Classifier", key="nav_cls", on_click=lambda: st.session_state.update(page="Classifier"))
    st.button("Patient Info", key="nav_pt", on_click=lambda: st.session_state.update(page="Patient Info"))

#  Header
if Path("assets/logo.png").exists():
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:0; margin-bottom:1.5rem">
            <img src="data:image/png;base64,{base64.b64encode(open('assets/logo.png','rb').read()).decode()}"
                 style="width:200px; margin-bottom:0.8rem"/>
            <h1 style="margin:0; font-size:2.4rem; font-weight:700; color:#0F172A">Welcome to OncoData</h1>
            <p style="margin:0; color:#475569">AI-Powered Cancer Image Classification</p>
            <p style="margin:0; color:#64748B; font-size:0.95rem">Upload medical images and analyse cancer types</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <h1 style="text-align:center; margin-top:0">OncoData</h1>
        <p style="text-align:center; color:#475569">AI-Powered Cancer Image Classification</p>
        <p style="text-align:center; color:#64748B">Upload medical images and analyse cancer types</p>
        """,
        unsafe_allow_html=True
    )


#  Router 
def show_home():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Choose Action")
    dest = st.selectbox("Choose where to go:", ["Classifier", "Patient Info"])
    if st.button("Go"):
        st.session_state.page = dest
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_classifier():
    # Header
    c_h1, c_h2, c_h3 = st.columns([1,2,1])
    with c_h2:
        st.markdown("<h3 style='text-align:center'>Upload Medical Image</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#64748B'>Image should be medical screening</p>", unsafe_allow_html=True)

    # Uploader card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload here",
        type=["jpg","jpeg","png","bmp","tif","tiff","webp"],
        accept_multiple_files=False,
        help="Limit 200MB per file • JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP",
        label_visibility="collapsed" 
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr />", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Please upload a medical image to run the classifier.")
        return

    # Back-to-home card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Go Back to Home"):
        st.session_state.page = "Home"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Check model files
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return
    if not os.path.isfile(LABELS_PATH):
        st.error(f"Label encoder not found at {LABELS_PATH}")
        return

    model   = load_model(MODEL_PATH)
    classes = load_classes(LABELS_PATH)

    # Preprocess
    image = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    x = (np.array(image).astype("float32") / 255.0)[None, ...]  # (1, H, W, 3)

    # Inference
    prob = model.predict(x, verbose=0)[0]
    idx  = int(prob.argmax())
    class_name = classes[idx]
    confidence = float(prob[idx])
    description = LABEL_DESCRIPTIONS.get(class_name, "No description available.")

    # Prediction card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        nice_name = pretty_label(class_name)  # <-- use pretty display name
        m1, m2 = st.columns(2)
        with m1: st.metric("Label", nice_name)
        with m2: st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(confidence)
        st.write(description)
    st.markdown("</div>", unsafe_allow_html=True)

def show_patient_info():
    st.markdown("<div class='card'><b>Patient Info</b><br/>Coming soon.</div>", unsafe_allow_html=True)

#  Router 
page = st.session_state.page
if page == "Home":
    show_home()
elif page == "Classifier":
    show_classifier()
elif page == "Patient Info":
    show_patient_info()
else:
    show_home()
# Universal footer
st.markdown(
    "<div style='text-align:center; color:#6B7280; font-size:13px; margin-top:40px'>"
    "© 2025 OncoData • Empowering medical diagnostics through intelligent technology"
    "</div>",
    unsafe_allow_html=True
)   