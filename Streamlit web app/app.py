import streamlit as st
import os
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
from pathlib import Path

# Load YOLOv8 model
model = YOLO("C:/Users/sp630/PycharmProjects/Weapoan detection web/.venv/best (12).pt")


# App title
st.title("Weapon Detection App (Gun & Knife)")

# File uploader
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

# ----------- IMAGE DETECTION FUNCTION -----------
def detect_image(image_path):
    image_path = Path(image_path)
    results = model.predict(source=str(image_path), save=True, conf=0.4)

    # Get the save directory
    save_dir = Path(results[0].save_dir)

    # Find the saved image in the directory
    for file in save_dir.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return file  # return path to detected image

    raise FileNotFoundError("Detected image not found in save directory.")

# ----------- VIDEO DETECTION FUNCTION -----------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video path
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", caption="Detecting...", use_column_width=True)

    cap.release()
    out.release()
    return temp_out.name

# ----------- MAIN STREAMLIT LOGIC -----------
if uploaded_file is not None:
    file_type = uploaded_file.type

    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    time.sleep(0.5)  # Small delay to ensure file is saved

    # If Image
    if "image" in file_type:
        st.image(temp_path, caption="Uploaded Image", use_column_width=True)
        st.success("Running Detection on Image...")
        output_path = detect_image(temp_path)
        img = Image.open(output_path)
        st.image(img, caption="Detected Image", use_column_width=True)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Detected Image", f, file_name="detected_image.jpg")

    # If Video
    elif "video" in file_type:
        st.video(temp_path)
        st.success("Running Detection on Video...")
        output_video_path = detect_video(temp_path)
        st.success("✅ Detection Complete!")

        with open(output_video_path, "rb") as f:
            st.download_button("⬇️ Download Detected Video", f, file_name="detected_video.mp4")
