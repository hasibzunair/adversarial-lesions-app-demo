"""Streamlit web app for melanoma detection"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import numpy as np
import time
import streamlit as st
import pandas as pd
from tensorflow import keras
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
    img_array = keras.preprocessing.image.load_img(
            uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    # Convert to array
    img_array = keras.preprocessing.image.img_to_array(img_array)
    # Normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array /= 255
    return img_array


model = cached_model()

if __name__ == '__main__':
    st.write("""
    # Hasib's AI Dermatology Assistant

    :red_circle: NOT FOR MEDICAL USE

    This is a prototype system for identifying presence of melanoma from skin lesions using neural networks.

    Publication: \n
    Zunair, Hasib, and A. Ben Hamza. 
    "Melanoma detection using adversarial training and deep transfer learning." 
    Physics in Medicine & Biology (2020), arxiv, https://arxiv.org/abs/2004.06824

    If you continue, you assume all liability when using the system.
    """)

    uploaded_file = st.file_uploader("Please upload a skin lesion image to to predict the presence of melanoma", type="jpg")

    if uploaded_file is not None:
        # Uploaded image
        original_image = keras.preprocessing.image.load_img(
            uploaded_file)

        st.image(original_image, caption="Input image of a skin lesion", use_column_width=True)
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
