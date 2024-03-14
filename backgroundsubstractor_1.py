import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

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
        
# Function to return a background subtractor
def get_bgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
    elif BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    elif BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True, varThreshold=100)
    elif BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    elif BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=15*60, isParallel=True)

st.title("Background Subtraction Demo")
bg_subtractor_choice = st.selectbox("Select Background Subtraction Method", ('GMG', 'MOG', 'MOG2', 'KNN', 'CNT'))
kernel_type = st.selectbox("Select Kernel Type", ('Ellipse', 'Rectangle', 'Cross'))
morph_type = st.selectbox("Select Morphological Operation", ('Closing', 'Opening', 'Dilation', 'Erosion', 'Combine'))
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False

if uploaded_file is not None and not st.session_state.processing_started:
    st.session_state.processing_started = True
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.getvalue())
    cap = cv2.VideoCapture(tfile.name)

    bg_subtractor = get_bgsubtractor(bg_subtractor_choice)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        bg_mask = bg_subtractor.apply(frame_resized)

        # Apply selected morphological operation
        bg_mask_filtered = get_filter(bg_mask, morph_type, kernel_type)
        res = cv2.bitwise_and(frame, frame, mask=bg_mask_filtered)
        
        # Convert mask to 3-channel BGR to display alongside the original frame
        bg_mask_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        combined_frame = np.hstack((frame_resized, bg_mask,  bg_mask_filtered,res))
        
        pil_img = Image.fromarray(combined_frame)
        stframe.image(pil_img, channels="BGR", use_column_width=True)

    cap.release()
    st.session_state.processing_started = False
