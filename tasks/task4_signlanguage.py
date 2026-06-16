import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

def count_fingers_and_detect_sign(image_np):
    """
    Classic OpenCV heuristic to detect hand gestures using skin color and convex hulls.
    Requires ZERO external ML dependencies, so it works flawlessly on Python 3.13.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "No Hand Detected", image_np
        
    # Get largest contour (assume it's the hand)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    
    if cv2.contourArea(cnt) < 3000:
        return "No Hand Detected", image_np
        
    # Draw contour
    cv2.drawContours(image_np, [cnt], -1, (0, 255, 0), 2)
    
    # Find convex hull and defects
    hull_points = cv2.convexHull(cnt)
    cv2.drawContours(image_np, [hull_points], -1, (255, 0, 0), 2)
    
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    
    sign = "Unknown Sign"
    try:
        defects = cv2.convexityDefects(cnt, hull_indices)
        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                
                # Cosine rule to find angle of the defect
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 57.2958
                
                # If angle <= 90 it means it's a gap between fingers
                if angle <= 90:
                    count += 1
                    cv2.circle(image_np, far, 5, [0, 0, 255], -1)
            
            fingers = count + 1
            if fingers == 1: sign = "One Finger (Attention)"
            elif fingers == 2: sign = "Peace / V"
            elif fingers >= 5: sign = "Open Palm (Hello / Stop)"
            else: sign = f"{fingers} Fingers"
    except Exception:
        pass
        
    return sign, image_np

def show_page():
    st.markdown('<div class="main-header">Sign Language Detection</div>', unsafe_allow_html=True)
    st.write("Description: Predicts sign language words in real-time or from an image. **Operational ONLY from 6 PM to 10 PM.**")

    current_hour = datetime.now().hour
    is_active_hours = 18 <= current_hour < 22
    
    st.markdown("---")
    override = st.checkbox("🔧 Developer Override (Disable Time Lock for testing)")
    
    if not is_active_hours and not override:
        st.error("🚫 Access Denied: This feature is currently locked.")
        st.info(f"The current time on your machine is {datetime.now().strftime('%I:%M %p')}. This feature is only operational between 6:00 PM and 10:00 PM.")
        st.stop()
        
    st.success("✅ Access Granted: Time check passed (or overridden).")
    st.markdown("---")

    input_mode = st.radio("Select Input Mode:", ["Live Webcam", "Upload Image"], horizontal=True)

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image of a hand gesture...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            sign, processed_img = count_fingers_and_detect_sign(image_rgb)
            
            st.image(processed_img, caption=f"Detected Sign: {sign}", use_container_width=True)
            st.success(f"Prediction: **{sign}**")

    elif input_mode == "Live Webcam":
        st.info("💡 Make sure you have good lighting and a plain background for the best detection!")
        run = st.checkbox("Turn On Webcam")
        FRAME_WINDOW = st.image([])
        
        if run:
            camera = cv2.VideoCapture(0)
            
            while run:
                success, frame = camera.read()
                if not success:
                    st.error("Failed to read frame from webcam.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sign, processed_img = count_fingers_and_detect_sign(frame_rgb)
                
                # Draw text on screen
                cv2.putText(processed_img, f"Sign: {sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                FRAME_WINDOW.image(processed_img)
            
            camera.release()
        else:
            st.write("Webcam is off.")
