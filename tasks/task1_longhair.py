import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

@st.cache_resource
def load_age_gender_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "best_model.keras")
        model = load_model(model_path)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_hair_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "hair_model.keras")
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        return None
    except Exception as e:
        return None

def preprocess_image_age_gender(uploaded_image):
    image = uploaded_image.convert("L")
    image = image.resize((128, 128), Image.BILINEAR)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 1)
    return image_array

def preprocess_image_hair(uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)
    return image_array

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

def show_page():
    st.markdown('<div class="main-header">Long Hair Identification</div>', unsafe_allow_html=True)
    
    st.write("Description: This task detects a person with long hair as female even if they are male, and a short-haired female as male. It operates exclusively for individuals aged between 20 and 30.")

    with st.spinner("Loading models..."):
        age_gender_model = load_age_gender_model()
        hair_model = load_hair_model()

    if age_gender_model is None:
        st.error("Age/Gender model not found. Please ensure best_model.keras is in the root directory.")
        return

    has_hair_model = hair_model is not None
    mock_hair_length = None
    if not has_hair_model:
        st.warning("⚠️ Hair Length model not found in `models/hair_model.keras`. Please train and download your model from Kaggle.")
        st.info("For now, you can manually mock the hair length below to test the age/gender logic override:")
        mock_hair_length = st.radio("Mock Hair Length:", ["Long", "Short"], horizontal=True)

    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("Analyze Images"):
        if uploaded_files:
            with st.spinner("Analyzing..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.container():
                        st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 1])

                        image = Image.open(uploaded_file)
                        col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)

                        processed_ag = preprocess_image_age_gender(image)
                        try:
                            predictions = age_gender_model.predict(processed_ag)
                            predicted_age = int(np.round(predictions[1][0]))
                            gender_prob = predictions[0][0]
                            original_gender = "Female" if gender_prob > 0.5 else "Male"
                        except Exception as e:
                            col2.error(f"Error during age/gender prediction: {e}")
                            continue
                        
                        hair_length = mock_hair_length
                        if has_hair_model:
                            processed_hair = preprocess_image_hair(image)
                            try:
                                hair_pred = hair_model.predict(processed_hair)[0][0]
                                hair_length = "Long" if hair_pred < 0.5 else "Short"
                            except Exception as e:
                                col2.error(f"Error during hair prediction: {e}")
                                hair_length = "Unknown"
                        
                        final_gender = original_gender
                        logic_applied = False
                        
                        if 20 <= predicted_age <= 30 and hair_length in ["Long", "Short"]:
                            logic_applied = True
                            if hair_length == "Long":
                                final_gender = "Female"
                            else:
                                final_gender = "Male"
                        
                        col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)
                        col2.markdown(f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {predicted_age}</div>', unsafe_allow_html=True)
                        
                        hair_color = "#38bdf8" if hair_length == "Short" else "#f472b6"
                        col2.markdown(f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(hair_color)))}, 0.1);">Hair Length: {hair_length} {"(Mocked)" if not has_hair_model else ""}</div>', unsafe_allow_html=True)

                        gender_color = "#9F7AEA" if final_gender == "Female" else "#4F46E5"
                        
                        if logic_applied:
                            col2.markdown(f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1); border: 2px solid #ef4444;">'
                                          f"Gender: {final_gender}<br>"
                                          f"<small style='color:#ef4444;'>* Overridden by Task 1 Logic (Age 20-30)*</small><br>"
                                          f"<small>Original Base Prediction: {original_gender}</small>"
                                          f"</div>", unsafe_allow_html=True)
                        else:
                            col2.markdown(f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                                          f"Gender: {final_gender}"
                                          f"</div>", unsafe_allow_html=True)
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("Please upload one or more images first.")
