# -*- coding: utf-8 -*-
# !/usr/bin/python
from tensorflow.keras.callbacks import Callback


class MonitorValLossCallback(Callback):
    """A custom tf.keras.callback class that monitors
    the validation loss value when the model stops training

    Args:
        Callback (tensorflow.keras.callbacks): tf.keras callback
    """

    def __init__(self):
        super(MonitorValLossCallback, self).__init__()

    def on_train_end(self, logs=None) -> None:
        """A built in tf.keras.callbacks function is called when training is finished.
        It raises a ValueError if validation loss does not satisfy a threshold.

        Args:
            logs ([dict], optional): Training logs. Defaults to None.

        Raises:
            ValueError
        """
        print("Finished")
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))
        print(logs.get("val_loss"))
        if logs.get("val_loss") > 0.7:
            raise ValueError
