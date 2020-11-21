import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.initializers import he_uniform
from configuration.config import Configuration


class Architecture():

    def __init__(self):
        config = Configuration().get_configuration()
        self.image_size = [config['training']['img_size_y'],
                           config['training']['img_size_x']]


    def __call__(self):
        
        print("Model architecture created")
        return self.get_sequential_model()

    def get_sequential_model(self):

        model = Sequential()

        model.add(BatchNormalization(input_shape=([*self.image_size, 3])))
        model.add(Conv2D(4, (3, 3), padding='valid',
                        #kernel_regularizer=tf.keras.regularizers.l1(0.04),
                        activation='relu',
                        kernel_initializer=he_uniform()))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(4, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(2, activation='softmax'))

        return model
