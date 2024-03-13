import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Define the get_kernel and get_filter functions as before

# Define the get_bgsubtractor function as before

# Streamlit UI components
st.title("Video Processing with Streamlit")

# Kernel type selection
selected_kernels = st.multiselect(
    "Select Kernel Types",
    options=["dilation", "opening", "closing", "combine"],
    default=["closing"]
)

# Background subtractor type selection
selected_bg_subtractor = st.selectbox(
    "Select Background Subtractor Type",
    options=["GMG", "MOG", "MOG2", "KNN", "CNT"]
)

# Video file uploader
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Process video if a file is uploaded
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    bg_subtractor = get_bgsubtractor(selected_bg_subtractor)

    stframe = st.empty()
    stvideo = st.empty()
    stvideo_filtered = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display original frame
        stframe.image(frame, channels="BGR")

        # Apply background subtractor
        bg_mask = bg_subtractor.apply(frame)

        # Display video after applying subtractor without filter
        stvideo.image(bg_mask, channels="GRAY")

        # Apply selected filters and display video
        for kernel in selected_kernels:
            filtered_frame = get_filter(bg_mask, kernel)
            stvideo_filtered.image(filtered_frame, channels="GRAY")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
