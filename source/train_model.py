# System libraries
import os
import sys

# Custom libraries
from configuration.config import Configuration
from computational_graph import training, inference

# Helper libraries
import tensorflow as tf
from tensorflow.compat.v1 import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# DEBUG, INFO, WARN, ERROR, FATAL
logging.set_verbosity(logging.WARN)

def status(opt_name, lr, image_size):

    print("Image size: {}".format(image_size))
    print("Optimizer: {}".format(opt_name))
    print("Learning Rate: {}".format(lr))


def main():

    config = Configuration().get_configuration()

    epochs = config['training']['epochs']
    lr = config['training']['lr']
    optimizer = config['training']['optimizer']
    img_size = [config['training']['img_size_x'],
                config['training']['img_size_y']]
    batch_size = config['training']['batch_size']
    model_save = config['training']['model_save']
    restore_model = config['training']['restore_model']

    status(optimizer, lr, img_size)
    print("Start Training...")
    t = training.Train(epochs, optimizer, lr, img_size, batch_size)

    if config['training']['training_method'] == "loop":
        t.train_loop(restore_model=restore_model, model_save=model_save)
    elif config['training']['training_method'] == "fit":
        t.train_fit(restore_model=restore_model, model_save=model_save)

    print("Start Inference..")
    infer = inference.Inference(batch_size, img_size)
    infer.predict()   


if __name__ == '__main__':

    main()
