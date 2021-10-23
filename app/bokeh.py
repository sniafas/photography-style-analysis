import argparse
import json
import gdown
import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import load_model


@st.cache(show_spinner=False)
def download_weights(model_choice):
    """
    Downloads model weights for deployment
    """

    # Create directory
    save_dest = Path("models")
    save_dest.mkdir(exist_ok=True)

    # Download weights for the chosen model
    if model_choice == "DenseNet (baseline)":
        url = "https://drive.google.com/uc?id=10-TWkCW_BAZLpGXkxPqXFV8lg-jnWNJD"
        output = "models/densenet.h5"

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == "VGG16 (baseline)":
        url = "https://drive.google.com/uc?id=1UaNIHQ-HYeN5v6egV9kAdwU0Nb4CfLBF"
        output = "models/vgg16.h5"

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == "DenseNet (best)":
        url = "https://drive.google.com/uc?id=1JUvuzyGQpScHyq2q25yhG962g3PMJ1eu"
        output = "models/densenet_best.h5"

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)

    elif model_choice == "VGG16 (best)":
        url = "https://drive.google.com/uc?id=19iu-Qhaofczgl6iMt6DSB_OHDBs9ggsr"
        output = "models/vgg16_best.h5"

        if not Path(output).exists():
            with st.spinner("Model weights were not found, downloading them. This may take a while."):
                gdown.download(url, output, quiet=False)
    else:
        raise ValueError("Unknown model: {}".format(model_choice))


def preprocess_image(image_file):
    """Preprocess image"""

    x, _ = process_path(image_file)
    x = np.expand_dims(x, axis=0)

    return x


def app_dof_predict(model_choice, image_file):

    # Download weights for the chosen model
    download_weights(model_choice)
    image = preprocess_image(image_file)
    prediction = {}

    if model_choice == "DenseNet (baseline)":
        model = load_model("models/densenet.h5", compile=False)
    elif model_choice == "VGG16 (baseline)":
        model = load_model("models/vgg16.h5", compile=False)
    elif model_choice == "DenseNet (best)":
        model = load_model("models/densenet_best.h5", compile=False)
    elif model_choice == "VGG16 (best)":
        model = load_model("models/vgg16_best.h5", compile=False)
    preds = model.predict(image)

    prediction = {
        "class": int(np.argmax(preds)),
        "probability": float(preds[0][np.argmax(preds)]),
    }

    return prediction


def decode_img(img):
    """Decode image and resize"""
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [200, 300])

    return img


def process_path(file_path):
    """Process input path"""
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, file_path


def plot_results(infer_images, inference_predicted_class, inference_predictions, class_names=["bokeh", "no bokeh"]):
    """Plot four images with predicted class and probabilities"""
    plt.figure(figsize=(40, 60))

    for i, (infer_img, _) in enumerate(infer_images.take(10)):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(infer_img.numpy() / 255)

        # Find the predicted class from predictions

        m = "Predicted: {}, {:.2f}%".format(class_names[inference_predicted_class[i]], inference_predictions[i] * 100)
        plt.title(m)
        plt.axis("off")
    plt.show()


def dof_predict(infer_images, model_path):

    trained_model = load_model(model_path, compile=False)

    inference_predicted_class = []
    inference_predictions = []
    results = {}
    for infer_img, img_name in infer_images:
        print(img_name)
        preds = trained_model.predict(tf.expand_dims(infer_img, axis=0))
        inference_predicted_class.append(np.argmax(preds))
        print(preds)
        inference_predictions.append(preds[0][np.argmax(preds)])

        results[str(img_name.numpy().decode("utf8").split("/")[-1])] = {
            "class": int(np.argmax(preds)),
            "prob": float(preds[0][np.argmax(preds)]),
        }

    plot_results(infer_images, inference_predicted_class, inference_predictions)

    return results


def save_results(results):
    """Save results to json"""
    json.dump(results, open("results.json", "w"))


def main(test_dir, model_path):

    # get the count of image files in the train directory
    inference_ds = tf.data.Dataset.list_files(test_dir + "/*", shuffle=False)

    infer_images = inference_ds.map(process_path)

    # inference
    results = dof_predict(infer_images, model_path)

    # save results
    save_results(results)


if __name__ == "__main__":

    # Initiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", action="store", help="Dataset path")
    parser.add_argument("-model", action="store", help="Model path")
    arguments = parser.parse_args()

    dataset = arguments.data
    model = arguments.model
    main(dataset, model)
