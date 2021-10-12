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
from tensorflow.keras.initializers import GlorotUniform

from src.configuration.ml_config import SEED


def bn_rl_conv(x, filters, kernel_size, layer_name):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=GlorotUniform(seed=SEED),
        padding="same",
        name=layer_name,
    )(x)
    return x


def dense_block(tensor, k, reps, rep_idx):
    for i in range(reps):
        x = bn_rl_conv(
            tensor,
            filters=4 * k,
            kernel_size=1,
            layer_name="dense_blk1_rep{}_{}".format(rep_idx, i),
        )
        x = bn_rl_conv(
            x,
            filters=k,
            kernel_size=3,
            layer_name="dense_blk2_rep{}_{}".format(rep_idx, i),
        )
        tensor = Concatenate()([tensor, x])
    return tensor


def transition_layer(x, theta, reps, rep_idx):
    f = int(backend.int_shape(x)[-1] * theta)
    x = bn_rl_conv(
        x,
        filters=f,
        kernel_size=1,
        layer_name="trans_layer_rep{}_{}".format(rep_idx, reps),
    )
    x = AvgPool2D(pool_size=2, strides=2, padding="same")(x)
    return x


def get_densenet(image_size, num_classes, filters, layer_reps):

    rep_idx = 1
    k = filters
    theta = 0.5
    repetitions = layer_reps

    inputs = Input(shape=[*image_size, 3])

    x = Conv2D(
        2 * k,
        7,
        strides=2,
        kernel_initializer=GlorotUniform(seed=SEED),
        padding="same",
        name="init_conv2d",
    )(inputs)
    x = MaxPool2D(3, strides=2, padding="same", name="init_maxp2d")(x)

    for reps in repetitions:
        d = dense_block(x, k, reps, rep_idx)
        x = transition_layer(d, theta, reps, rep_idx)
        rep_idx += 1

    x = GlobalAvgPool2D()(d)

    output = Dense(num_classes, kernel_initializer=GlorotUniform(seed=SEED), activation="softmax")(x)
    model = Model(inputs, output, name="DenseNet")

    return model
