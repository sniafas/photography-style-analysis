# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
from src.configuration.ml_config import SEED


def set_seeds():

    # DEBUG, INFO, WARN, ERROR, FATAL
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import random
    import numpy as np
    import tensorflow as tf

    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    print("Static seeds are set")


def set_device(device):

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    if device == "gpu":
        config = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        config.gpu_options.per_process_gpu_memory_fraction = 0.75
        config.gpu_options.allow_growth = True
        InteractiveSession(config=config)

    elif device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
