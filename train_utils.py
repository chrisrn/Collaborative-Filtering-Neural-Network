# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


# model selection

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# dl libraraies
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, merge
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ReduceLROnPlateau

from keras.layers.merge import dot
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

# specifically for deeplearning.
from keras.layers import Dropout, Flatten, Activation, Input, Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
import random as rn
import json
import os


# Class for learning rate view during training
class LRlogs(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """
        Function for printing the learning rate during training
        :param epoch: int epoch number
        :param logs: str for stdout
        :return: None
        """

        lr = K.eval(self.model.optimizer.lr)
        print('learning_rate: {}'.format(lr))


class ModelHandler(object):
    def __init__(self,
                 batch_size,
                 epochs,
                 learning_rate,
                 n_users,
                 n_books,
                 adaptive_lr,
                 adaptive_lr_patience_epochs,
                 adaptive_lr_decay,
                 min_adaptive_lr,
                 early_stopping,
                 early_stopping_min_change,
                 early_stopping_patience_epochs,
                 save_model,
                 model_dir,
                 epochs_per_save,
                 exponential_lr,
                 num_epochs_per_decay,
                 lr_decay_factor,
                 gpus):
        """
        Initialization function for all the parameters related to model
        :param batch_size: int batch size
        :param epochs: int number of epochs
        :param learning_rate: float learning rate
        :param n_users: int unique number of users
        :param n_books: int unique number of books
        :param adaptive_lr: boolean indicator for using adaptive learning rate callback
        :param adaptive_lr_patience_epochs: int number of patience epochs until the decrease of lr
        :param adaptive_lr_decay: float factor of lr decrease
        :param min_adaptive_lr: float minimun lr to reach
        :param early_stopping: boolean indicator for using early stopping callback
        :param early_stopping_min_change: float minimum delta between min val_loss and current val_loss
        :param early_stopping_patience_epochs: int number of patience epochs until the end of training
        :param save_model: boolean indicator for saving the model into .hdf5 file
        :param model_dir: str path to save the model
        :param epochs_per_save: int number of epochs per model save
        :param exponential_lr: boolean indicator for using exponential learning rate callback
        :param num_epochs_per_decay: int number of epochs per lr decay
        :param lr_decay_factor: float factor of lr decrease
        :param gpus: int number of gpus
        """

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_users = n_users
        self.n_books = n_books
        self.adaptive_lr = adaptive_lr
        self.early_stopping = early_stopping
        self.save_model = save_model
        self.model_dir = model_dir
        self.exponential_lr = exponential_lr
        self.adaptive_lr_patience_epochs = adaptive_lr_patience_epochs
        self.adaptive_lr_decay = adaptive_lr_decay
        self.min_adaptive_lr = min_adaptive_lr
        self.early_stopping_min_change = early_stopping_min_change
        self.early_stopping_patience_epochs = early_stopping_patience_epochs
        self.epochs_per_save = epochs_per_save
        self.num_epochs_per_decay = num_epochs_per_decay
        self.lr_decay_factor = lr_decay_factor
        self.gpus = gpus

    def get_callbacks(self):
        """
        Function for filling the keras callbacks list
        :return: list with keras callbacks
        """

        callbacks = []

        if self.adaptive_lr:
            print('***** Adaptive Learning Rate activated *****')
            patience = self.adaptive_lr_patience_epochs
            factor = self.adaptive_lr_decay
            min_lr = self.min_adaptive_lr
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                          patience=patience, min_lr=min_lr)
            callbacks.append(reduce_lr)

        if self.early_stopping:
            print('***** Early Stopping activated *****')
            min_delta = self.early_stopping_min_change
            patience = self.early_stopping_patience_epochs
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                                           patience=patience, verbose=0,
                                                           mode='auto', baseline=None,
                                                           restore_best_weights=False)
            callbacks.append(early_stopping)

        if self.save_model:
            period = self.epochs_per_save
            filepath = os.path.join(self.model_dir, "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5")
            ckpt = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                   save_best_only=False,
                                                   save_weights_only=False,
                                                   mode='auto', period=period)
            callbacks.append(ckpt)

        if self.exponential_lr:
            print('***** Exponential Learning Rate decay activated *****')
            def schedule(epoch, lr):
                epochs_per_decay = self.num_epochs_per_decay
                decay_factor = self.lr_decay_factor
                if epoch % epochs_per_decay == 0 and epoch != 0:
                    return lr * decay_factor
                else:
                    return lr
            exp_lr = keras.callbacks.LearningRateScheduler(schedule)
            callbacks.append(exp_lr)

        callbacks.append(LRlogs())

        return callbacks

    def get_model(self):
        """
        Function for model graph construction
        :return: keras model object
        """

        n_latent_factors_user = 15
        n_latent_factors_book = 20
        n_latent_factors_mf = 8
        n_users, n_books = self.n_users, self.n_books

        book_input = keras.layers.Input(shape=[1], name='Item')
        book_embedding_mlp = keras.layers.Embedding(n_books + 1, n_latent_factors_book,
                                                     name='Book-Embedding-MLP')(book_input)
        book_vec_mlp = keras.layers.Flatten(name='FlattenBooks-MLP')(book_embedding_mlp)
        book_vec_mlp = keras.layers.Dropout(0.2)(book_vec_mlp)

        book_embedding_mf = keras.layers.Embedding(n_books + 1, n_latent_factors_mf, name='Book-Embedding-MF')(
            book_input)
        book_vec_mf = keras.layers.Flatten(name='FlattenBooks-MF')(book_embedding_mf)
        book_vec_mf = keras.layers.Dropout(0.2)(book_vec_mf)

        user_input = keras.layers.Input(shape=[1], name='User')
        user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(
            keras.layers.Embedding(n_users + 1, n_latent_factors_user, name='User-Embedding-MLP')(user_input))
        user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)

        user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(
            keras.layers.Embedding(n_users + 1, n_latent_factors_mf, name='User-Embedding-MF')(user_input))
        user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

        concat = keras.layers.Concatenate()([book_vec_mlp, user_vec_mlp])
        concat_dropout = keras.layers.Dropout(0.2)(concat)
        dense = keras.layers.Dense(200, name='FullyConnected')(concat_dropout)
        dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
        dropout_1 = keras.layers.Dropout(0.2, name='Dropout-1')(dense_batch)
        dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(dropout_1)
        dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)

        dropout_2 = keras.layers.Dropout(0.2, name='Dropout-2')(dense_batch_2)
        dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dropout_2)
        dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

        pred_mf = dot([book_vec_mf, user_vec_mf], name='Dot', axes=1)

        pred_mlp = keras.layers.Dense(1, activation='relu', name='Activation')(dense_4)

        combine_mlp_mf = keras.layers.Concatenate()([pred_mf, pred_mlp])
        result_combine = keras.layers.Dense(100, name='Combine-MF-MLP')(combine_mlp_mf)
        deep_combine = keras.layers.Dense(100, name='FullyConnected-4')(result_combine)

        result = keras.layers.Dense(1, name='Prediction')(deep_combine)

        nn_model = keras.Model([user_input, book_input], result)

        if self.gpus:
            nn_model = multi_gpu_model(nn_model, gpus=self.gpus)

        # plot_model(nn_model, to_file='cf_nn.png', show_shapes=True, show_layer_names=True)
        nn_model.summary()
        nn_model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_absolute_error')

        return nn_model


def plot_loss(History):
    """
    Function for plotting train-validation loss
    :param History: keras model fit object
    """

    plt.plot(History.history['loss'], 'g')
    plt.plot(History.history['val_loss'], 'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.grid(True)
    plt.show()
