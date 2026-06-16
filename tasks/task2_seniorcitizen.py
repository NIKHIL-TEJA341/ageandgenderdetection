import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import time
import collections

try:
    from keras.models import load_model
except ImportError:
    st.error("Please install tensorflow/keras.")

@st.cache_resource
def load_age_gender_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "best_model.keras")
        model = load_model(model_path)
        return model
    except Exception as e:
        return None

def preprocess_face(face_image):
    image = Image.fromarray(face_image).convert("L")
    image = image.resize((128, 128), Image.BILINEAR)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 1)
    return image_array

def log_to_csv(age, gender):
    file_path = "visitor_logs.csv"
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    new_data = pd.DataFrame([{
        "Date": date_str,
        "Time": time_str,
        "Age": age,
        "Gender": gender,
        "Category": "Senior Citizen" if age > 60 else "Regular"
    }])
    
    if not os.path.exists(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)

def show_page():
    st.markdown('<div class="main-header">Senior Citizen Identification</div>', unsafe_allow_html=True)
    st.write("Description: Predicts multiple persons in real-time webcam feed. If a person's age is > 60, marks them as a senior citizen and logs data to an Excel/CSV file.")
    
    with st.spinner("Loading Age & Gender Model..."):
        model = load_age_gender_model()
        
    if model is None:
        st.error("Age/Gender model not found. Please ensure best_model.keras is in the root directory.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Live Camera Feed")
        run = st.checkbox("Turn On Webcam")
        FRAME_WINDOW = st.image([])
        
    with col2:
        st.markdown("### Visitor Logs")
        log_placeholder = st.empty()
        
        def update_log_view():
            if os.path.exists("visitor_logs.csv"):
                with open("visitor_logs.csv", "r") as f:
                    header = f.readline().strip().split(',')
                    last_lines = list(collections.deque(f, maxlen=10))
                    if last_lines:
                        last_lines = [line.strip().split(',') for line in last_lines]
                        df = pd.DataFrame(last_lines, columns=header)
                        log_placeholder.dataframe(df)
                    else:
                        log_placeholder.info("No logs yet.")
            else:
                log_placeholder.info("No logs yet.")
                
        update_log_view()
        
        st.info("💡 Data is automatically logged to `visitor_logs.csv` when a face is detected. A 5-second cooldown prevents spamming the file for the same person.")

    if "last_log_time" not in st.session_state:
        st.session_state.last_log_time = 0

    if run:
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Could not access the webcam. Please ensure it is connected and not being used by another app.")
            run = False
            return

        while run:
            success, frame = camera.read()
            if not success:
                st.error("Failed to read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                face_crop = frame_rgb[y:y+h, x:x+w]
                
                try:
                    processed_face = preprocess_face(face_crop)
                    predictions = model.predict(processed_face, verbose=0)
                    
                    predicted_age = int(np.round(predictions[1][0]))
                    gender_prob = predictions[0][0]
                    predicted_gender = "Female" if gender_prob > 0.5 else "Male"
                    
                    is_senior = predicted_age > 60
                    
                    box_color = (255, 0, 0) if is_senior else (0, 255, 0)
                    label = f"Senior! Age:{predicted_age} {predicted_gender}" if is_senior else f"Age:{predicted_age} {predicted_gender}"
                    
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(frame_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                    current_time = time.time()
                    if current_time - st.session_state.last_log_time > 5.0:
                        log_to_csv(predicted_age, predicted_gender)
                        st.session_state.last_log_time = current_time
                        update_log_view()
                        
                except Exception as e:
                    pass

            FRAME_WINDOW.image(frame_rgb)

        camera.release()
    else:
        st.write("Webcam is off.")
