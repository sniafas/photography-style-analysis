"""Streamlit web app for depth of field detection"""

import time
from PIL import Image
import streamlit as st
from bokeh import app_dof_predict

from tempfile import NamedTemporaryFile

temp_file = NamedTemporaryFile(delete=False)

# Page layout
st.set_page_config(page_title="Depth of Field Detection", page_icon=":camera:", layout="wide")

# Sidebar options
st.sidebar.title("Prediction Settings")
st.sidebar.text("")

models = ["DenseNet (baseline)", "VGG16 (baseline)", "DenseNet (best)", "VGG16 (best)"]

model_choice = []
st.sidebar.write("Choose a model for prediction")
model_choice.append(st.sidebar.radio("", models))

with st.container():
    st.title("Depth of Field detection w/ Deep Learning")
    st.image(
        "https://source.unsplash.com/mj2NwYH3wBA/960x640",
        use_column_width="auto",
    )

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if file is not None:
        img = Image.open(file)
        temp_file.write(file.getvalue())
        st.image(img, caption="Uploaded image", use_column_width="auto")

        if st.button("Predict"):
            st.write("")
            st.write("Working...")

            start_time = time.time()

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
