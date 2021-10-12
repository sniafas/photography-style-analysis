# -*- coding: utf-8 -*-
# !/usr/bin/python

from tensorflow.keras import backend
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    ReLU,
    Conv2D,
    Dense,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool2D,
    Concatenate,
    MaxPooling2D,
    Flatten,
    Dropout,
)
from tensorflow.keras.initializers import GlorotUniform, HeUniform

from src.configuration.ml_config import SEED


def bn_rl_conv(x, filters, kernel_size, strides, padding):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=HeUniform(seed=SEED),
        padding=padding,
    )(x)
    return x


def get_simplenet(image_size, num_classes, filters):

    inputs = Input(shape=[*image_size, 3])

    x = bn_rl_conv(inputs, filters, 12, 1, "valid")
    x = MaxPool2D(2, padding="valid")(x)

    x = bn_rl_conv(x, filters, 5, 1, "valid")
    x = bn_rl_conv(x, filters, 5, 1, "valid")
    x = MaxPool2D(2, padding="valid")(x)

    x = bn_rl_conv(x, filters * 2, 3, 1, "valid")
    x = bn_rl_conv(x, filters * 2, 3, 1, "valid")
    x = MaxPool2D(2, padding="valid")(x)

    # mlp
    # x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, kernel_initializer=HeUniform(seed=SEED), activation="relu")(x)
    x = Dropout(0.2)(x)

    # output
    output = Dense(num_classes, kernel_initializer=HeUniform(seed=SEED), activation="softmax")(x)
    model = Model(inputs, output, name="SimpleNet")

    return model
