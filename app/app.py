"""Streamlit web app for depth of field detection"""

import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from bokeh import app_dof_predict

from tempfile import NamedTemporaryFile

temp_file = NamedTemporaryFile(delete=False)


# ---------------------------------#

# Page layout
# Page expands to full width

st.set_page_config(page_title="Depth of Field Detection", layout="wide")


# ---------------------------------#

# Sidebar options
st.sidebar.title("Prediction Settings")
st.sidebar.text("")

models = ["DenseNet (baseline)", "VGG16 (baseline)", "DenseNet (best)", "VGG16 (best)"]

model_choice = []
st.sidebar.write("Choose a model for prediction")
model_choice.append(st.sidebar.radio("", models))

# ---------------------------------#

# Main Page options

col1, col2, col3 = st.columns([1, 6, 1])


with col2:
    st.title("Depth of Field detection w/ Deep Learning")

with col2:
    st.image(
        "https://source.unsplash.com/iEmpY2HvOeU/960x640",
        caption="An example input w/ shallow dof (Bokeh).",
        width=800,
    )

with col2:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

with col2:

    if file is not None:
        img = Image.open(file)
        temp_file.write(file.getvalue())
        st.image(img, caption="Uploaded image", width=300)

        if st.button("Predict"):
            st.write("")
            st.write("Working...")

            start_time = time.time()
            scores = np.zeros((1, 7))

            for choice in model_choice:
                prediction = app_dof_predict(choice, temp_file.name)
                print(prediction)
                execute_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.001)
                    execute_bar.progress(percent_complete + 1)
            prob = prediction["probability"]
            if prediction["class"] == 0:
                st.header("Prediction: Bokeh - Confidence {:.1f}%".format(prob * 100))
            elif prediction["class"] == 1:
                st.header("Prediction: No bokeh detected - Confidence {:.1f}%".format(prob * 100))

            st.write("Took {} seconds to run.".format(round(time.time() - start_time, 2)))

# ---------------------------------#
