import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("🚗 Deteksi Kendaraan CCTV Realtime")

source = st.selectbox("Pilih sumber video:", ["Webcam", "File Lokal", "CCTV Link (m3u8)"])

if source == "Webcam":
    video_source = 0
elif source == "File Lokal":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        video_source = uploaded_file.name
        with open(video_source, "wb") as f:
            f.write(uploaded_file.read())
    else:
        st.stop()
elif source == "CCTV Link (m3u8)":
    url = st.text_input("https://pelindung.bandung.go.id:3443/video/HIKSVISION/Soekar.m3u8")
    if url:
        video_source = url
    else:
        st.stop()

cap = cv2.VideoCapture(video_source)

frame_window = st.image([])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.4, verbose=False)
    annotated = results[0].plot()

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    frame_window.image(annotated_rgb)

cap.release()


