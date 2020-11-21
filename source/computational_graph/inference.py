import tensorflow as tf
from utils.train_utils import load_dataset, get_optimizer, get_true_labels, get_naming
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np

class Inference:

    def __init__(self, batch_size, image_size):

        self.batch_size = batch_size
        self.model = load_model('trained_models/model_'+get_naming()+'.h5')

    def predict(self):

        test, test_len = load_dataset('test', self.batch_size, 3000, method='loop')

        test_preds = self.model.predict(
            test, steps=test_len / self.batch_size)

        test_y = get_true_labels(test, test_len, self.batch_size)

        print(confusion_matrix(test_y, np.argmax(
            test_preds, axis=1), labels=(0, 1, 2)))
