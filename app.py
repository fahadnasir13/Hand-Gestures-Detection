import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Load  YOLO model
model = YOLO("handgesture.pt")


st.title("🖐️ Hand Gesture Detection App")
st.write("Upload an image of a hand gesture to see what it means.")

# Upload image
uploaded_file = st.file_uploader("Choose a hand gesture image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Run YOLO model
    results = model.predict(image_np, conf=0.5)[0]

    # Draw bounding boxes and labels
    annotated_frame = image_np.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

        # Draw rectangle and label
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

       

    # Show annotated image
    st.image(annotated_frame, caption="Detected Gesture", channels="RGB")
