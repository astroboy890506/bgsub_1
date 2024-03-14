import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tempfile

# Function to return a background subtractor
def get_bgsubtractor(BGS_TYPE):
    # https://docs.opencv.org/3.4/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120,
                                                        decisionThreshold = 0.8)
    # https://docs.opencv.org/3.4/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures = 5,
                                                       backgroundRatio = 0.7, noiseSigma=0)
    # https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows = True,
                                                  varThreshold = 100)
    # https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold=400,
                                                 detectShadows = True)
    # https://docs.opencv.org/3.4/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,
                                                        useHistory = True,
                                                        maxPixelStability = 15*60,
                                                        isParallel=True)

st.title("Background Subtraction Demo")
bg_subtractor_type = st.selectbox("Select Background Subtraction Method", ('GMG', 'MOG', 'MOG2', 'KNN', 'CNT'))
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

# Initialize session state
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False

if uploaded_file is not None and not st.session_state.processing_started:
    st.session_state.processing_started = True
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.getvalue())
    cap = cv2.VideoCapture(tfile.name)

    bg_subtractor_type = get_bgsubtractor(bg_subtractor_type)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Apply background subtraction
        bg_mask = bg_subtractor_type.apply(frame_resized)
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
