import numpy as np
import tensorflow as tf
import importlib
import datetime

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean
import preprocess.tf_records as tfr
import matplotlib.pyplot as plt
from matplotlib import rc

from configuration.config import Configuration
from augmentation.augmentation import normalization, augment_with_flip, augment_with_rotate, augment_patches, patches_random
from architectures.model_architectures import Architecture
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD, Adadelta, Nadam

from tensorflow.keras.models import load_model


def load_dataset(data_type, batch_size, prefetch_size, method, normalize=True, augmentation=False):
    """
    Load tf record dataset, augmentation compatible

    Arguments:
        data_type: String type of dataset (train,valid,test)
        batch_size: Int, batch size
        prefetch_size: Integerprefetch buffer, int
        augmentation: boolean

    Returns:
        dataset, dataset_length
    """

    tfrio = tfr.TFRecordIO()
    dataset, dataset_len = tfrio.read_tf_record(data_type)

    dataset = dataset.shuffle(1000)
    #print("Shuffle")    
    if normalize:
        print("Normalization...")
        dataset = dataset.map(normalization)

    if augmentation:
        print("Augmentation enabled")
        dataset = dataset.map(patches_random, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #dataset = dataset.shuffle(1000).prefetch(PREFETCH_SIZE).repeat().batch(BATCH_SIZE)
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)#.prefetch(prefetch_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if method == 'fit':
        dataset = dataset.repeat()

    #image = tf.squeeze(image)
    #image_patches = tf.reshape(image[:,:,:], [9, 200, 300, 3])

    return dataset, dataset_len

def get_optimizer(learning_rate, opt_name):
    '''
    Select optimizer method
    :param learning_rate: lr values
    :param opt_name: optimizer name
    adam, nadam, rms, adagrad, sgd, adadelta
    :return: optimizer object
    '''
    if opt_name == 'adam':
        optimizer = Adam(learning_rate)
    elif opt_name == 'nadam':
        optimizer = Nadam(learning_rate)
    elif opt_name == 'rms':
        optimizer = RMSprop(learning_rate)
    elif opt_name == 'adagrad':
        optimizer = Adagrad(learning_rate)
    elif opt_name == 'sgd':
        optimizer = SGD(learning_rate)
    elif opt_name == 'adadelta':
        optimizer = Adadelta(learning_rate)

    return optimizer

def get_true_labels(dataset, dataset_len, batch_size):

    true_y = []
    for _, y in dataset.take(dataset_len // batch_size + 1):
        true_y.append(np.argmax(y, axis=1))

    return np.concatenate(true_y)


def losses_and_metrics():

    loss_fn = CategoricalCrossentropy()
    train_loss = Mean(name='train_loss')
    train_accuracy = CategoricalAccuracy('train_accuracy')

    valid_loss = Mean(name='valid_loss')
    valid_accuracy = CategoricalAccuracy('valid_accuracy')

    return loss_fn, train_loss, train_accuracy, valid_loss, valid_accuracy


def write_summary(train_loss, train_accuracy, valid_loss, valid_accuracy, epoch): 

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'source/logs/tboard_logs/' + current_time + '/train'
    valid_log_dir = 'source/logs/tboard_logs/' + current_time + '/test'
    self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

def train_ckpt_manager(optimizer, model, save_path='source/logs/tf_ckpts'):
    '''
    Method that creates tf.train.Checkpoint and tf.train.CheckpointManager used for model saving during training process

    Args:
        optimizer: Object, tf optimizer
        model: Object, tf model
    
    Return:
        ckpt: Object, tf.train.Checkpoint
        manager: Object, tf.train.CheckpointManager
    '''

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=1)

    return ckpt, manager


def model_initialise():
    '''
    Returns a Sequantial model
    '''
    arch_obj = Architecture()
    model = arch_obj()

    return model


def plot_training(results):

    config = Configuration().get_configuration()
    epochs = config['training']['epochs']

    plt.style.use(['dark_background', 'bmh'])
    rc('figure', figsize=(8, 8), max_open_warning=False)
    rc('axes', facecolor='none')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(results['accuracy'])
    ax1.plot(results['val_accuracy'])
    #mpl.pyplot.xticks(np.arange(1, 2, step=1))  # Set label locations.
    ax1.set_xlim(1, epochs-1)
    ax1.set_xticks(np.arange(0, epochs))        # set xtick values
    ax1.set_xticklabels(np.arange(1, epochs+1))   # set sparse xtick values
    ax1.grid()
    ax1.set(ylabel="Accuracy")
    ax1.legend(['Training Accuracy', 'Validation Accuracy'])

    ax2.plot(results['loss'])
    ax2.plot(results['val_loss'])
    #mpl.pyplot.xticks(np.arange(1, 2, step=1))  # Set label locations.
    ax2.set_xlim(1, epochs-1)
    ax2.set_xticks(np.arange(0, epochs))        # set xtick values
    ax2.set_xticklabels(np.arange(1, epochs+1))   # set sparse xtick values
    ax2.grid()

    ax2.set(xlabel='Epochs')
    ax2.set(ylabel="Loss")
    ax2.legend(['Training Loss', 'Validation Loss'])
    plt.savefig("plots/results_"+get_naming()+".png")


def get_timestamp():
    return datetime.datetime.now().strftime("%y-%m-%d-%H:%M")

def get_naming():

    config = Configuration().get_configuration()
    optimizer = config['training']['optimizer']
    size = config['training']['img_size_x']
    return "{}_{}".format(size, optimizer)
