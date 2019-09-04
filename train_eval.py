from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from keras.models import load_model
import json
import pandas as pd
import os
import numpy as np
from math import sqrt

from train_utils import plot_loss, ModelHandler
from data_utils import DataHandler

import warnings
# Turn off warnings
warnings.filterwarnings('ignore')


def data_processing(args):
    """
    Function for processing and splitting the data
    :param args: dictionary with parameters from params.json file
    :return: dataframes from train-test-validation sets and 2 integes with unique number of users and books
    """

    print('Data processing started...')

    data_handler = DataHandler(args['data_dir'],
                               args['clean_titles'],
                               args['debug_data'],
                               args['debug_rows'],
                               args['test_split'],
                               args['validation_split'],
                               args['multiple_isbn_pickle'])

    dataset_file = os.path.join(args['data_dir'], 'dataset.csv')
    if args['create_data']:
        ratings, books, users = data_handler.read_data()

        users = data_handler.users_processing(users)
        books = data_handler.books_processing(books)
        ratings = data_handler.ratings_processing(ratings)

        dataset = data_handler.get_dataset(users, books, ratings)
        dataset.to_csv(dataset_file, sep=';')
    else:
        dataset = pd.read_csv(dataset_file, sep=';')
        del dataset['Unnamed: 0']

    train, test, validation = data_handler.split_data(dataset)
    print(train.head())

    print('Train rows: {}'.format(len(train)))
    print('Validation rows: {}'.format(len(validation)))
    print('Test rows: {}'.format(len(test)))

    num_unique_users = len(dataset['user_id'].unique())
    num_unique_books = len(dataset['unique_isbn'].unique())

    print('Data processing ended...')

    return train, test, validation, num_unique_users, num_unique_books


def main():
    """
    Main function
    """

    with open('params.json') as json_file:
        args = json.load(json_file)

    train_set, test_set, validation_set, n_users, n_books = data_processing(args)

    model_handler = ModelHandler(args['batch_size'],
                                 args['epochs'],
                                 args['learning_rate'],
                                 n_users,
                                 n_books,
                                 args['adaptive_learning_rate'],
                                 args['adaptive_lr_patience_epochs'],
                                 args['adaptive_lr_decay'],
                                 args['min_adaptive_lr'],
                                 args['early_stopping'],
                                 args['early_stopping_min_change'],
                                 args['early_stopping_patience_epochs'],
                                 args['save_model'],
                                 args['model_dir'],
                                 args['epochs_per_save'],
                                 args['exponential_lr'],
                                 args['num_epochs_per_decay'],
                                 args['lr_decay_factor'],
                                 args['gpus'])

    callbacks = model_handler.get_callbacks()

    print('Start training...')
    if args['fine-tuning-file']:
        print('Fine-tuning from {}'.format(args['fine-tuning-file']))
        model = load_model(args['fine-tuning-file'])
    else:
        model = model_handler.get_model()

    if not args['forward_pass']:
        history = model.fit([train_set['user_id'], train_set['unique_isbn']], train_set['book_rating'],
                            batch_size=args['batch_size'],
                            epochs=args['epochs'],
                            validation_data=([validation_set['user_id'], validation_set['unique_isbn']], validation_set['book_rating']),
                            verbose=1,
                            callbacks=callbacks)
        print('End of training...')

        plot_loss(history)

    # Predictions on test set
    predictions = model.predict([test_set['user_id'], test_set['unique_isbn']])
    meanAbsoluteError = mean_absolute_error(test_set['book_rating'],
                                            predictions)
    meanSquaredError = mean_squared_error(test_set['book_rating'],
                                          predictions)
    if args['log_test_predictions']:
        predictions = predictions.round().astype('int64')
        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_df = pd.concat([test_set, predictions_df], axis=1)
        print(predictions_df.head(20))

    print("MAE on test set: {}".format(meanAbsoluteError))
    print("MSE on test set: {}".format(meanSquaredError))
    print("RMSE on test set: {}".format(sqrt(meanSquaredError)))


if __name__ == "__main__":
    main()
