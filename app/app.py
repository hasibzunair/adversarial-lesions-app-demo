"""Streamlit web app for melanoma detection"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import numpy as np
import time
import cv2
import streamlit as st
import pandas as pd
from tensorflow import keras
from PIL import Image
from melanet.pretrained_model import get_model

# Model package lives at:
# https://github.com/hasibzunair/adversarial-lesions/tree/master/packaging 
# https://test.pypi.org/project/melanet/

st.set_option("deprecation.showfileUploaderEncoding", False)

IMAGE_SIZE = 256


@st.cache(allow_output_mutation=True)
def cached_model():
    model = get_model()
    model._make_predict_function()
    return model

def preprocess_image(uploaded_file):
    # Load image
    img_array = np.array(Image.open(uploaded_file))
    # Convert to array
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    # Normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array /= 255
    return img_array


model = cached_model()

if __name__ == '__main__':
    st.write("""
    # Hasib's AI Dermatology Assistant

    :red_circle: NOT FOR MEDICAL USE!

    This is a prototype system for predicting the presence of melanoma from dermoscopic skin lesions using neural networks.

    Publication: \n
    Zunair, Hasib, and A. Ben Hamza. 
    "Melanoma detection using adversarial training and deep transfer learning." 
    Physics in Medicine & Biology (2020), arxiv, https://arxiv.org/abs/2004.06824

    Made with :heart:, by [Hasib Zunair](https://hasibzunair.github.io/).

    If you continue, you assume all liability when using the system.

    Please upload a [dermoscopic](https://dermnetnz.org/topics/dermoscopy/) skin lesion image to predict the presence of melanoma. Here's an example.
    """)

    example_image = np.array(Image.open("media/example.jpg"))
    st.image(example_image, caption="An example input.", width=100)

    uploaded_file = st.file_uploader("Upload file by browzing or drag and drop the image here.", type="jpg")

    if uploaded_file is not None:
        # Uploaded image
        original_image = np.array(Image.open(uploaded_file))

        st.image(original_image, caption="Input image of the skin lesion", use_column_width=True)
        st.write("")
        st.write("Analyzing the input image. Please wait...")

        start_time = time.time()

        # Preprocess input image
        image = preprocess_image(uploaded_file)
        image = np.expand_dims(image, 0)
        # Predict
        predictions = model.predict(image)
        nonmelanoma = predictions[0][0] * 100
        melanoma = predictions[0][1] * 100

        st.write("Took {} seconds to run.".format(
            round(time.time() - start_time, 3)))
        # Show predictions
        source = pd.DataFrame({
            'Skin Lesion Type': ['Non-melanoma', 'Melanoma', ],
            'Confidence Score': [nonmelanoma, melanoma],
        })

        st.dataframe(source)
