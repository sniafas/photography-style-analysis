# -*- coding: utf-8 -*-
# !/usr/bin/python

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
)


def get_vgg_model(image_size, num_classes):
    """Get VGG16 model"""
    inputs = Input(shape=[*image_size, 3])
    model = VGG16(
        include_top=False,
        weights="imagenet",
        classes=num_classes,
        input_tensor=inputs,
        classifier_activation=None,
    )

    model.trainable = False

    # Route model's embeddings to avg pooling
    x = GlobalAveragePooling2D(name="avg_pooling")(model.output)
    x = BatchNormalization()(x)

    outputs = Dense(num_classes, activation="softmax", name="preds")(x)

    model = Model(inputs, outputs, name="VGG16")

    return model
