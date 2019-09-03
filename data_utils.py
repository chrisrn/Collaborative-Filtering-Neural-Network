import numpy as np
import pandas as pd
import os
import csv
import pickle

from sklearn.model_selection import train_test_split


class DataHandler(object):
    def __init__(self,
                 data_dir,
                 clean_titles,
                 debug_data,
                 debug_rows,
                 test_split,
                 validation_split,
                 multiple_isbn_pickle):
        self.data_dir = data_dir
        self.clean_titles = clean_titles
        self.debug_data = debug_data
        self.debug_rows = debug_rows
        self.test_split = test_split
        self.validation_split = validation_split
        self.multiple_isbn_pickle = multiple_isbn_pickle

        self.ratings_file = os.path.join(self.data_dir, 'BX-Book-Ratings.csv')
        self.users_file = os.path.join(self.data_dir, 'BX-Users.csv')
        self.books_file = os.path.join(self.data_dir, 'BX-Books.csv')
        self.books_new = os.path.join(self.data_dir, 'BX-Books-new.csv')

    def rewrite_books(self):
        with open(self.books_file, 'rb') as f:
            reader = csv.reader(f, delimiter=";")
            lines = list(reader)

        with open(self.books_new, 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            for line in lines:
                if len(line) > 8:
                    for i in range(1, 1+len(line)-8):
                        line[1] += line[2]
                        line.remove(line[2])
                writer.writerow(line)

    def read_data(self):

        if self.clean_titles:
            self.rewrite_books()

        ratings = pd.read_csv(self.ratings_file, sep=';')
        books = pd.read_csv(self.books_new, sep=';')
        users = pd.read_csv(self.users_file, sep=';')

        if self.debug_data:
            ratings = ratings.loc[:self.debug_rows, :]
            books = books.loc[:self.debug_rows, :]
            users = users.loc[:self.debug_rows, :]

        return ratings, books, users

    def users_processing(self, users):
        # Change column names for our needs
        users.columns = users.columns.str.strip().str.lower().str.replace('-', '_')
        # Convert unrealistic ages to NaN
        users.loc[(users.age < 5) | (users.age > 100), 'age'] = np.nan

        # We split the Location field into City, State, Country
        user_location_expanded = users.location.str.split(',', 2, expand=True)
        user_location_expanded.columns = ['city', 'state', 'country']
        users = users.join(user_location_expanded)
        users.drop(columns=['location'], inplace=True)
        users.country.replace('', np.nan, inplace=True)

        return users

    def books_processing(self, books):

        # Drop unused columns
        books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)
        books.columns = books.columns.str.strip().str.lower().str.replace('-', '_')

        # Convert years to float
        books.year_of_publication = pd.to_numeric(books.year_of_publication, errors='coerce')

        # Replace all years of zero with NaN
        books.year_of_publication.replace(0, np.nan, inplace=True)

        # Remove books with unrealistic year of publication
        old_books = books[books.year_of_publication < 1900]
        future_books = books[books.year_of_publication > 2018]
        books = books.loc[~(books.isbn.isin(old_books.isbn))]
        books = books.loc[~(books.isbn.isin(future_books.isbn))]

        return books

    def ratings_processing(self, ratings):

        ratings.columns = ratings.columns.str.strip().str.lower().str.replace('-', '_')

        # Remove zero ratings for having only explicit ratings in range of 1 to 10
        ratings = ratings[ratings.book_rating != 0]

        return ratings

    def make_isbn_dict(self, df, has_mult_isbns):
        title_isbn_dict = {}
        for title in has_mult_isbns.index:
            isbn_series = df.loc[df.book_title == title].isbn.unique()
            title_isbn_dict[title] = isbn_series.tolist()
        return title_isbn_dict

    def get_dataset(self, users, books, ratings):

        books_with_ratings = ratings.join(books.set_index('isbn'), on='isbn')

        # Remove rows with missing title
        books_with_ratings.dropna(subset=['book_title'], inplace=True)

        multiple_isbns = books_with_ratings.groupby('book_title').isbn.nunique()
        has_mult_isbns = multiple_isbns.where(multiple_isbns > 1)
        has_mult_isbns.dropna(inplace=True)  # NaNs removal means removal of books with only one ISBN

        pickle_file = os.path.join(self.data_dir, 'multiple_isbn_dict.pickle')
        if self.multiple_isbn_pickle:
            dict_unique_isbn = self.make_isbn_dict(books_with_ratings, has_mult_isbns)
            with open(pickle_file, 'wb') as handle:
                pickle.dump(dict_unique_isbn, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(pickle_file, 'rb') as handle:
            multiple_isbn_dict = pickle.load(handle)

        books_with_ratings['unique_isbn'] = books_with_ratings.apply(
            lambda row: multiple_isbn_dict[row.book_title][0] if row.book_title in multiple_isbn_dict.keys() else row.isbn,
            axis=1)

        books_users_ratings = books_with_ratings.join(users.set_index('user_id'), on='user_id')
        dataset = books_users_ratings[['user_id', 'unique_isbn', 'book_rating']]

        dataset['user_id'] = dataset['user_id'].astype('category').cat.codes.values
        dataset['unique_isbn'] = dataset['unique_isbn'].astype('category').cat.codes.values

        return dataset

    def split_data(self, dataset):
        train, test = train_test_split(dataset, test_size=self.test_split)
        train, validation = train_test_split(train, test_size=self.validation_split)

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        validation.reset_index(drop=True, inplace=True)
        return train, test, validation
