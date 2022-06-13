import io

import requests
import streamlit as st
from PIL import Image
from loguru import logger


def load_image(image_file):
    img = Image.open(io.BytesIO(image_file))
    return img


def main():
    st.title("Mask Classification")

    menu = ["Image", "Video"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            img_byte_arr = image_file.read()
            files = {"image": img_byte_arr}
            response = requests.post("http://api:5000/detect_image", files=files)
            if response.status_code == 200:
                logger.info(response.content)
                # To View Uploaded Image
                st.image(load_image(response.content), width=250)


if __name__ == "__main__":
    main()
