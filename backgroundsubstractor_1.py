import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to return a background subtractor
def get_bgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

# Streamlit app layout
st.title("Background Subtraction Demo")

# Upload video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

# Placeholder for the video frames
frame_placeholder = st.empty()

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV video capture object
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
    
    # Initialize background subtractor
    bg_subtractor = get_bgsubtractor('KNN')

    # Function to process each frame
    def process_frame(frame):
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Apply background subtraction
        bg_mask = bg_subtractor.apply(frame_resized)
        # Convert mask to 3-channel BGR (to display alongside the original frame)
        bg_mask_bgr = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
        # Stack both frames horizontally for display
        combined_frame = np.hstack((frame_resized, bg_mask_bgr))
        return combined_frame

    # Process and display the video
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames to read

            # Process the frame
            processed_frame = process_frame(frame)

            # Convert to PIL Image to display in Streamlit
            pil_img = Image.fromarray(processed_frame)

            # Update the placeholder with the new image
            frame_placeholder.image(pil_img, channels="BGR")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    cap.release()
else:
    st.text("Upload a video file to get started.")
