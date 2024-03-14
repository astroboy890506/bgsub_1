import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Existing get_kernel function
def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'Ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    elif KERNEL_TYPE == 'Rectangle':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    elif KERNEL_TYPE == 'Cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    else:
        kernel = np.ones((3,3), np.uint8)  # Default case, for safety
    return kernel

# Modified get_filter function to use user-selected kernel
def get_filter(img, filter, kernel_type):
    kernel = get_kernel(kernel_type)
    if filter == 'Closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif filter == 'Opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    elif filter == 'Dilation':
        return cv2.dilate(img, kernel, iterations=2)
    elif filter == 'Erosion':
        return cv2.erode(img, kernel, iterations=2)
    elif filter == 'Combine':
        # Example sequence: Close -> Open -> Dilate
        step1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        step2 = cv2.morphologyEx(step1, cv2.MORPH_OPEN, kernel, iterations=2)
        final = cv2.dilate(step2, kernel, iterations=2)
        return final

# Existing get_bgsubtractor function...

st.title("Background Subtraction Demo")
bg_subtractor_type = st.selectbox("Select Background Subtraction Method", ('GMG', 'MOG', 'MOG2', 'KNN', 'CNT'))
kernel_type = st.selectbox("Select Kernel Type", ('Ellipse', 'Rectangle', 'Cross'))
morph_type = st.selectbox("Select Morphological Operation", ('Closing', 'Opening', 'Dilation', 'Erosion', 'Combine'))
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

# Existing session state initialization...

if uploaded_file is not None and not st.session_state.processing_started:
    st.session_state.processing_started = True
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.getvalue())
    cap = cv2.VideoCapture(tfile.name)

    bg_subtractor = get_bgsubtractor(bg_subtractor_type)  # Variable name changed for clarity

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        bg_mask = bg_subtractor.apply(frame_resized)

        # Apply selected morphological operation
        bg_mask_filtered = get_filter(bg_mask, morph_type, kernel_type)

        # Convert mask to 3-channel BGR to display alongside the original frame
        bg_mask_bgr = cv2.cvtColor(bg_mask_filtered, cv2.COLOR_GRAY2BGR)
        combined_frame = np.hstack((frame_resized, bg_mask_bgr))
        
        pil_img = Image.fromarray(combined_frame)
        stframe.image(pil_img, channels="BGR", use_column_width=True)

    cap.release()
    st.session_state.processing_started = False
