# -*- coding: utf-8 -*-
# !/usr/bin/python

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten, Dropout
from tensorflow.keras.initializers import GlorotUniform
from src.configuration.ml_config import SEED


def get_lenet(image_size, num_classes, filters=20):

    inp = Input(shape=[*image_size, 3])

    x = Conv2D(
        filters=20,
        kernel_size=5,
        strides=1,
        kernel_initializer=GlorotUniform(seed=SEED),
    )(inp)
    x = Dropout(0.5, seed=SEED)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(
        filters=50,
        kernel_size=5,
        strides=1,
        kernel_initializer=GlorotUniform(seed=SEED),
    )(x)
    x = Dropout(0.5, seed=SEED)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation="relu")(x)
    x = Dropout(0.5, seed=SEED)(x, training=True)
    x = Dense(num_classes, activation="softmax")(x)

    return Model(inp, x, name="lenet-all")
