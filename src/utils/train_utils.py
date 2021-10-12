import json
import numpy as np
import pandas as pd
import tensorflow as tf
import importlib
import datetime

import tensorflow_addons as tfa
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, Precision, Recall, AUC

import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD, Adadelta, Nadam
from tensorflow.keras.models import load_model


from src.configuration.config import Configuration
from src.transformations.basic import normalization, decode_img, oh_label
from src.architectures.model_architectures import Architecture


def dataset_to_tensor(data, batch_size, shuffle=False, batch=True, mode="sl"):
    """Convert a dataframe in tensor dataset"""
    # convert csv records to tensors
    image_path = tf.convert_to_tensor(data["photo_id"], dtype=tf.string)

    if mode == "sl":  # supervised train(baseline dataset)
        labels = tf.convert_to_tensor(data["label"])
        # create a tensor dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
        # num_parallel_calls=8
        dataset = dataset.map(map_dataset, num_parallel_calls=1)

    elif mode == "al":  # inference
        dataset = tf.data.Dataset.from_tensor_slices((image_path))
        dataset = dataset.map(map_inference_dataset, num_parallel_calls=1)

    if shuffle:
        dataset = dataset.shuffle(
            tf.data.experimental.cardinality(dataset).numpy() * 3,
            reshuffle_each_iteration=False,
        )

    if batch:
        dataset = dataset.batch(batch_size).prefetch(1000)

    return dataset


def read_dataset_from_csv(data_type, path):
    """Read dataset from csv

    Args:
        data_type (str): train/valid/test

    Returns:
        pd: data
    """
    data = pd.read_csv(tf.io.gfile.glob(path + data_type + "*")[0])

    return data


def map_dataset(img_path, label):
    """
    Returns:
        image, label
    """
    config = Configuration().get_configuration()
    prefix = config["dataset"]["img_path"]

    # path/label represent values for a single example
    image_file = tf.io.read_file(prefix + img_path)
    image = decode_img(image_file)
    label = oh_label(label)

    return image, label


def map_inference_dataset(path):
    """We don't know the label, we need image and path to know which image to annotate

    Returns:
        image, path
    """
    config = Configuration().get_configuration()
    prefix = config["dataset"]["img_path"]

    # path/label represent values for a single example
    image_file = tf.io.read_file(prefix + path)
    image = decode_img(image_file)

    return image, path


def active_data_splits(train, valid, pool_dataset, split_ratio, mode):
    """Slice and concatenate simulation dataset for active learning

    Args:
        train ([type]): training
        valid ([type]): validation
        pool_dataset ([type]): pool dataset
        split_ratio ([type]): split ratio
    """
    # stepping addition from 100 to 2000 samples
    if mode == "random":
        pool_dataset = pool_dataset.sample(pool_dataset.shape[0], random_state=0)
    slice_dataset = pool_dataset[:split_ratio]

    # take a copy of static dataset
    td = train.copy()
    vd = valid.copy()

    # concatenate w/ the temporal dataset by 80/20 split
    td = pd.concat([td, slice_dataset[: int(len(slice_dataset) * 80 / 100)]])
    vd = pd.concat([vd, slice_dataset[int(len(slice_dataset) * 80 / 100) :]])

    return td, vd


def model_initialise():
    """
    Returns:
        tf.keras.Model
    """
    arch = Architecture()

    return arch()


def get_optimizer(learning_rate, opt_name):
    """Select optimizer method

    Arguments:
        learning_rate: float, learning value
        opt_name: str, optimizer name (adam, nadam, rms, adagrad, sgd, adadelta)

    Returns:
        optimizer object
    """
    if opt_name == "adam":
        optimizer = Adam(learning_rate)
    elif opt_name == "nadam":
        optimizer = Nadam(learning_rate)
    elif opt_name == "rms":
        optimizer = RMSprop(learning_rate)
    elif opt_name == "adagrad":
        optimizer = Adagrad(learning_rate)
    elif opt_name == "sgd":
        optimizer = SGD(learning_rate, nesterov=True)
    elif opt_name == "adadelta":
        optimizer = Adadelta(learning_rate)

    return optimizer


def get_true_labels(dataset, dataset_len, batch_size):

    true_y = []
    for _, y in dataset.take(dataset_len // batch_size + 1):
        true_y.append(np.argmax(y, axis=1))

    return np.concatenate(true_y)


def losses_and_metrics(num_classes):
    """
    Define loss and metrics
    Loss: Categorical Crossentropy
    Metrics: (Train, Validation) Accuracy, Precision, Recall, F1

    Returns:
        loss, train loss, train acc, valid loss, valid acc, precision, recall, auc, f1
    """

    loss_fn = CategoricalCrossentropy(from_logits=True)
    train_loss = Mean(name="train_loss")
    train_accuracy = CategoricalAccuracy("train_accuracy")

    valid_loss = Mean(name="valid_loss")
    valid_accuracy = CategoricalAccuracy("valid_accuracy")

    precision = Precision(name="precision")
    recall = Recall(name="recall")
    auc = AUC(name="auc")
    f1_train = tfa.metrics.F1Score(num_classes=num_classes, average="macro")
    f1_loss = tfa.metrics.F1Score(num_classes=num_classes, average="macro")

    return (
        loss_fn,
        train_loss,
        train_accuracy,
        valid_loss,
        valid_accuracy,
        precision,
        recall,
        auc,
        f1_train,
        f1_loss,
    )


def plot_training(results, path_to_save):

    config = Configuration().get_configuration()
    epochs = config["training"]["epochs"]
    title = "Training History"
    plt.style.use(["dark_background", "bmh"])
    rc("figure", figsize=(5, 8), max_open_warning=False)
    rc("axes", facecolor="none")
    figure = plt.figure(figsize=(8, 5), facecolor="white")

    plt.title(title, {"fontname": "Roboto", "fontsize": 15})
    plt.xlabel("Epochs", {"fontname": "Roboto", "fontsize": 12})
    plt.ylabel("Accuracy", {"fontname": "Roboto", "fontsize": 12})
    plt.plot(results["acc"])
    plt.plot(results["val_acc"])
    plt.xlim((0, epochs - 1))
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Training", "Validation"])

    plt.savefig(f"{path_to_save}_acc.png")

    figure = plt.figure(figsize=(8, 5), facecolor="white")

    plt.title(title, {"fontname": "Roboto", "fontsize": 15})
    plt.xlabel("Epochs", {"fontname": "Roboto", "fontsize": 12})
    plt.ylabel("Loss", {"fontname": "Roboto", "fontsize": 12})
    plt.plot(results["loss"])
    plt.plot(results["val_loss"])
    plt.xlim((0, epochs - 1))
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Training", "Validation"])

    plt.savefig(f"{path_to_save}_loss.png")

    figure = plt.figure(figsize=(8, 5), facecolor="white")

    plt.title(title, {"fontname": "Roboto", "fontsize": 15})
    plt.xlabel("Epochs", {"fontname": "Roboto", "fontsize": 12})
    plt.ylabel("F1", {"fontname": "Roboto", "fontsize": 12})
    plt.plot(results["f1_score"])
    plt.plot(results["val_f1_score"])
    plt.xlim((0, epochs - 1))
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Training", "Validation"])

    plt.savefig(f"{path_to_save}_f1.png")
