import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import time

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh;
    }
    [data-testid="stImage"] img {
        display: block;
        margin: auto;
        max-height: 85vh;
    }
    </style>
""", unsafe_allow_html=True)

model = YOLO("yolov8s.pt")

st.title("🚗 Deteksi Kemacetan Realtime (Anti-Scroll Mode)")

source_option = st.selectbox("Pilih sumber video:", ["Webcam", "Upload Video", "Link CCTV (m3u8)"])
video_source = None

if source_option == "Webcam":
    video_source = 0
elif source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_source = temp_file.name
elif source_option == "Link CCTV (m3u8)":
    url = st.text_input("Masukkan link stream (format .m3u8):", 
                        value="https://pelindung.bandung.go.id:3443/video/HIKSVISION/Soekar.m3u8")
    if url:
        video_source = url

if video_source is not None:
    cap = cv2.VideoCapture(video_source)

    col1, col2 = st.columns([4, 1])
    with col1:
        video_placeholder = st.empty() 
    with col2:
        status_placeholder = st.empty()

    stop_button = st.button("Stop", key="stop-button")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Gagal membaca frame video.")
            break

        frame = cv2.resize(frame, (960, 540))

        results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.4, verbose=False)
        annotated = results[0].plot()

        class_ids = results[0].boxes.cls.cpu().numpy()
        vehicle_count = len([cid for cid in class_ids if cid in [2, 3, 5, 7]])

        if vehicle_count <= 10:
            status = "LANCAR 🟢"
            color = "green"
        elif vehicle_count <= 20:
            status = "PADAT 🟡"
            color = "orange"
        else:
            status = "MACET 🔴"
            color = "red"

        status_placeholder.markdown(
            f"<div style='font-size:22px;color:{color};font-weight:bold;'>"
            f"🚗 Jumlah: {vehicle_count}<br>🚦 Status: {status}</div>",
            unsafe_allow_html=True
        )

        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        time.sleep(0.01)

    cap.release()
    st.success("✅ Deteksi selesai.")
else:
    st.info("Silakan pilih sumber video dulu.")
