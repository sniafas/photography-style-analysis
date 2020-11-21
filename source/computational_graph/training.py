# System libraries
import os
import sys
import importlib
import datetime

# Custom libraries
from configuration.config import Configuration
import preprocess.tf_records as tfr
from utils.gpu_util import set_gpu
from utils.train_utils import load_dataset, get_optimizer, get_true_labels, losses_and_metrics, train_ckpt_manager, model_initialise, plot_training, get_naming
from utils.training_callbacks import early_stopping, model_chpnt

# Helper libraries
import tensorflow as tf
from tensorflow.compat.v1 import logging
import numpy as np
from collections import defaultdict


class Train:

    def __init__(self, epochs, opt_name, lr, image_size, batch_size):

        set_gpu('gpu')

        self.epochs = epochs
        self.lr = lr
        self.optimizer = get_optimizer(lr, opt_name)
        self.batch_size = batch_size
        self.image_size = image_size
        
        config = Configuration().get_configuration()
        self.normalize = config['training']['normalize']
        self.augment = config['training']['augment']


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'source/logs/tboard_logs/' + current_time + '/training'
        valid_log_dir = 'source/logs/tboard_logs/' + current_time + '/validation'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)        

    def train_fit(self, restore_model=False, model_save=True):

        model = model_initialise()
        ckpt, ckpt_manager = train_ckpt_manager(self.optimizer, model)      

        if restore_model:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))

        if model_save:
            callbacks = [model_chpnt(),early_stopping()]
        else:
            callbacks = [early_stopping()]


        train, train_len = load_dataset('train', self.batch_size, 1000, normalize=self.normalize, augmentation=self.augment, method='fit')
        valid, valid_len = load_dataset('valid', self.batch_size, 3000, method='fit')

        print("Data loaded")
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model compiled")

        results = model.fit(train, validation_data=valid,
                                epochs=self.epochs,
                                steps_per_epoch=train_len // self.batch_size,
                                validation_steps=valid_len // self.batch_size,
                                callbacks=callbacks)

        plot_training(results.history)
        if model_save:
            model.save('trained_models/model_'+get_naming()+'.h5')

    def train_loop(self, restore_model=False, model_save = True):

        results = defaultdict(list)
        min_loss = 100
        
        model = model_initialise()
        ckpt, ckpt_manager = train_ckpt_manager(self.optimizer, model)
        loss_fn, train_loss, train_accuracy, valid_loss, valid_accuracy = losses_and_metrics()        


        if restore_model:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))

        # Initialise summarization variables for tensorboard
        def _write_summary(train_loss, train_accuracy, valid_loss, valid_accuracy, epoch): 

            with self.train_summary_writer.as_default(): 
                tf.summary.scalar('Loss', train_loss, step=epoch)
                tf.summary.scalar('Accuracy', train_accuracy, step=epoch)
                
            with self.valid_summary_writer.as_default():
                tf.summary.scalar('Loss', valid_loss, step=epoch)
                tf.summary.scalar('Accuracy', valid_accuracy, step=epoch)     


        @tf.function
        def training_graph(x, y, training):

            with tf.GradientTape() as tape:
                predictions = model(x, training=training)
                loss = loss_fn(y, predictions)

            if training:

                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_accuracy(y, predictions)

            else:
                valid_loss(loss)
                valid_accuracy(y, predictions)

        
        train, train_len =  load_dataset('train', self.batch_size, 1000, normalize=self.normalize, augmentation=self.augment, method='loop')
        valid, _ = load_dataset('valid', self.batch_size, 3000, method='loop')
        pb = tf.keras.utils.Progbar(train_len//self.batch_size, stateful_metrics='val_loss')

        # Training-Validation process
        for epoch in range(self.epochs):

            print("Epoch {}/{}".format(epoch+1,self.epochs))

            for i, (x, y) in enumerate(train):

                training_graph(x, y, True)
                pb.update(i+1)

            for x, y in valid:
                training_graph(x, y, False)

            train_status_msg = "Training Loss: {:.3f}, Training Acc: {:.2f}% | Valid Loss: {:.3f}, Valid Acc: {:.2f}%"
            print(train_status_msg.format(train_loss.result(), train_accuracy.result()*100,
                                          valid_loss.result(), valid_accuracy.result()*100))

            if valid_loss.result() < min_loss:
                min_loss = valid_loss.result()
                save_path = ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("Loss {:1.2f}".format(min_loss))

            # Keep results for plotting
            results['loss'].append(train_loss.result().numpy())
            results['accuracy'].append(train_accuracy.result().numpy())
            results['val_loss'].append(valid_loss.result().numpy())
            results['val_accuracy'].append(valid_accuracy.result().numpy())
            

            _write_summary(train_loss.result(),
                        train_accuracy.result(),
                        valid_loss.result(),
                        valid_accuracy.result(),
                        epoch)

            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()

        plot_training(results)
        if model_save:
            model.save('trained_models/model_'+get_naming()+'.h5')
