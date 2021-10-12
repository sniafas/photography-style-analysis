# -*- coding: utf-8 -*-
# !/usr/bin/python

from src.configuration.config import Configuration
from src.architectures.dense_net import get_densenet
from src.architectures.lenet import get_lenet
from src.architectures.simplenet import get_simplenet
from src.architectures.vgg_net import get_vgg_model

# from tensorflow.keras.mixed_precision import experimental as mixed_precision

# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)


class Architecture:
    def __init__(self):

        config = Configuration().get_configuration()

        self.image_size = [
            config["training"]["img_size_y"],
            config["training"]["img_size_x"],
        ]
        self.num_classes = config["training"]["num_classes"]
        self.model_arch = config["training"]["model_arch"]
        self.filters = config["training"]["filters"]
        self.reps = config["training"]["repetitions"]

    def __call__(self):

        print("Model architecture created")
        print(self.num_classes)
        return self.get_model_arch(self.model_arch, self.num_classes)

    def get_model_arch(self, arch_name, num_classes):

        if arch_name == "simplenet":
            return self._get_simplenet(num_classes)

        elif arch_name == "densenet":
            return self._get_densenet_model(num_classes)

        elif arch_name == "lenet":
            return self._get_lenet_model(num_classes)

        elif arch_name == "vgg":
            return get_vgg_model(self.image_size, num_classes)

    def _get_densenet_model(self, num_classes):
        return get_densenet(self.image_size, num_classes, self.filters, self.reps)

    def _get_lenet_model(self, num_classes):
        return get_lenet(self.image_size, num_classes)

    def _get_simplenet(self, num_classes):
        return get_simplenet(self.image_size, num_classes, self.filters)
