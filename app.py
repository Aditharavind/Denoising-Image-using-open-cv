#Image Denoising open-cv median-blur 
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

# Streamlit App Title
st.title("Image Denoising using Median Filter")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img_cv = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

    # Show original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Slider for median filter kernel size
    ksize = st.slider("Select Median Filter Kernel Size (Odd Number)", 1, 15, 3, step=2)

    # Apply median blur filter
    denoised_image = cv2.medianBlur(img_cv, ksize)

    # Show the processed image
    st.image(denoised_image, caption="Denoised Image", use_column_width=True)

    # Button to download the processed image
    if st.button("Download Denoised Image"):
        img_pil = Image.fromarray(denoised_image)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img_pil.save(temp_file.name)
        st.download_button("Download", temp_file.name, "denoised_image.png")

