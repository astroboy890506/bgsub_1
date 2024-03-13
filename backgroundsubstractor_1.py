import streamlit as st
import numpy as np
import cv2
from random import randint
import tempfile

def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    elif KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    elif KERNEL_TYPE == 'closing':
        kernel = np.ones((3,3), np.uint8)
    else:
        kernel = None
    return kernel

def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    elif filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations = 2)
    elif filter == 'dilation':
        return cv2.dilate(img, get_kernel('dilation'), iterations = 2)
    elif filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations = 2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations = 2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation
    return img # Return the image as is if the filter type does not match

# Background subtractor function omitted for brevity - remains unchanged

def main():
    st.title("Background Subtraction and Filtering")

    # Selectors for kernel and BGS type
    kernel_type = st.multiselect('Select Kernel Type', ['dilation', 'opening', 'closing', 'combine'], default=['closing'])
    bgs_type = st.selectbox('Select Background Subtractor Type', ['GMG', 'MOG', 'MOG2', 'KNN', 'CNT'])

    # Video file uploader
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        bg_subtractor = get_bgsubtractor(bgs_type)

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % 20 == 0:  # Process and display every 20th frame for simplicity
                frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
                bg_mask = bg_subtractor.apply(frame)

                for k_type in kernel_type:
                    filtered_frame = get_filter(bg_mask, k_type)
                    st.image(filtered_frame, channels="BGR", caption=f"Frame {frame_number} with {k_type} filter")

            frame_number += 1

        cap.release()

if __name__ == "__main__":
    main()
