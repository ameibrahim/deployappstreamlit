import streamlit as st
from ultralytics import YOLO
import cv2

st.title("YOLOv8 Inference")

image_file = st.file_uploader("Upload image", type=["jpg", "png", "svg"])

if image_file:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(image_file.read())

    model = YOLO("yolov8n.pt")  # model in same folder
    result = model("uploaded_image.jpg")[0].plot()
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    st.image(result_rgb, caption="Prediction", use_container_width=True)