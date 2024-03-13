import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tempfile

# Function to return a background subtractor
def get_bgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

st.title("Background Subtraction Demo")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

# Initialize session state
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False

if uploaded_file is not None and not st.session_state.processing_started:
    st.session_state.processing_started = True
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.getvalue())
    cap = cv2.VideoCapture(tfile.name)

    bg_subtractor = get_bgsubtractor('KNN')

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Apply background subtraction
        bg_mask = bg_subtractor.apply(frame_resized)
        # Convert mask to 3-channel BGR (to display alongside the original frame)
        bg_mask_bgr = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
        # Stack both frames horizontally for display
        combined_frame = np.hstack((frame_resized, bg_mask_bgr))
        
        # Convert to PIL Image and display
        pil_img = Image.fromarray(combined_frame)
        stframe.image(pil_img, channels="BGR", use_column_width=True)

    cap.release()
    # Reset the flag to allow re-processing if another file is uploaded
    st.session_state.processing_started = False
