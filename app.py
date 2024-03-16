import streamlit as st
import os
import cv2
import pandas as pd
from tamp_detect import preprocess,ocr_pan,plot_bounding_boxes,ocr_text



def main():
    st.title("Pan Card Text Detection using OCR")

    uploaded_file = st.sidebar.file_uploader("Upload PAN Card Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        global image_path
        image_path = os.path.join("uploads", uploaded_file.name)

        # st.subheader("Preprocessing Image...")
        preprocessed_image = preprocess(image_path)

        # st.subheader("Extracting Text...")
        df = ocr_pan(preprocessed_image)

        st.subheader("Bounding Boxes Plot")
        image_with_boxes = plot_bounding_boxes(df, image_path)
        
        # st.subheader("Preprocessing Extracted Text...")
        df_new = ocr_text(df)
        st.write(df_new)

        os.remove(image_path)
        st.sidebar.success("Uploaded image deleted successfully.")

if __name__ == "__main__":
    main()