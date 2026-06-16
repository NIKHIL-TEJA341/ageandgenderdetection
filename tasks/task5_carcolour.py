import streamlit as st
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

def detect_blue_color(image_np):
    """
    Detects if the dominant color in the car bounding box is blue.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    blue_pixels = cv2.countNonZero(mask)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    
    if total_pixels == 0:
        return False
    
    blue_ratio = blue_pixels / total_pixels
    return blue_ratio > 0.05

def process_image(image_np, model):

    results = model(image_np)
    
    car_count = 0
    person_count = 0
    
    output_image = image_np.copy()
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            
  
            if conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls == 0:
                    person_count += 1
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                elif cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    car_count += 1
                    
                    car_roi = image_np[y1:y2, x1:x2]
                    
                    if car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
                        is_blue = detect_blue_color(car_roi)
                        
                        if is_blue:

                            color = (255, 0, 0) 
                            label = "Blue Car"
                        else:
                            # Other cars -> Blue rectangle (RGB: 0, 0, 255)
                            color = (0, 0, 255)
                            label = "Other Car"
                            
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output_image, car_count, person_count

def show_page():
    st.markdown('<div class="main-header">Car Colour & Counting</div>', unsafe_allow_html=True)
    st.write("Description: Predicts car colours in traffic. Counts cars and people. Shows a red rectangle for blue cars, and blue for others.")
    
    if not HAS_YOLO:
        st.error("Please install the ultralytics package to run YOLOv8 object detection.")
        st.code("pip install ultralytics", language="bash")
        return
        
 
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt')
        
    with st.spinner("Loading YOLOv8 model for object detection..."):
        model = load_model()
        
    uploaded_file = st.file_uploader("Upload an image of a traffic scene...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        

        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        with st.spinner("Processing image..."):
            processed_img, cars, people = process_image(image_np, model)
            
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(processed_img, caption="Processed Traffic Scene", use_container_width=True)
            
        with col2:
            st.markdown("### Detection Results")
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #1E3A8A;">Total Cars: {cars}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #1E3A8A;">Total People: {people}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("🚗 **Blue cars** are highlighted with a **RED** rectangle.\n\n🚙 **Other cars** are highlighted with a **BLUE** rectangle.\n\n🚶 **People** are highlighted in **GREEN**.")
