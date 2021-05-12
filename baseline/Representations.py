import os
import sys

import numpy as np
import pandas as pd
import psutil
from sktime.transformers.series_as_features.rocket import Rocket
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from framework.Dataset import Dataset


class TSFreshRepresentation():

    def __init__(self, config: Configuration, dataset: Dataset):
        self.config = config
        self.dataset = dataset

    def create_representation(self):
        x_train, x_val, x_test = self.dataset.x_train, self.dataset.x_test_val, self.dataset.x_test
        features = list(dataset.feature_names_all)

        x_train = self.transform(x_train, features)
        x_val = self.transform(x_val, features)
        x_test = self.transform(x_test, features)

        x_train = extract_features(x_train, column_id="id", column_sort="time", n_jobs=40)
        x_val = extract_features(x_val, column_id="id", column_sort="time", n_jobs=40)
        x_test = extract_features(x_test, column_id="id", column_sort="time", n_jobs=40)
        x_train, x_val, x_test = impute(x_train), impute(x_val), impute(x_test)

        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        np.save(self.dataset.dataset_folder + self.config.ts_fresh_features_train, x_train)
        np.save(self.dataset.dataset_folder + self.config.ts_fresh_features_val, x_val)
        np.save(self.dataset.dataset_folder + self.config.ts_fresh_features_test, x_test)

        print('Representation generation finished.')
        print('Resulting shapes of train, val and test: ', x_train.shape, x_val.shape, x_test.shape)

    def transform(self, np_array, features):
        """
        Transforms the dataset from a np_array into a dataframe that is expected by TS Fresh

        :param np_array: The dataset that should be transformed into a dataframe
        :param features: The names of the features in the third dimension of the np_array
        :return: The resulting dataframe
        """

        # https://stackoverflow.com/questions/36235180/efficiently-creating-a-pandas-dataframe-from-a-numpy-3d-array
        m, n, r = np_array.shape
        timestamps = [i for i in range(n)] * m
        out_arr = np.column_stack((np.repeat(np.arange(m), n), np_array.reshape(m * n, -1)))
        cols = ['id'] + features
        df = pd.DataFrame(out_arr, columns=cols)
        df['id'] = df['id'].astype(int)
        df['time'] = timestamps
        return df

    def load_into_dataset(self):
        self.dataset.x_train = np.load(self.dataset.dataset_folder + self.config.ts_fresh_features_train)
        self.dataset.x_test_val = np.load(self.dataset.dataset_folder + self.config.ts_fresh_features_val)
        self.dataset.x_test = np.load(self.dataset.dataset_folder + self.config.ts_fresh_features_test)

        self.dataset.time_series_length = None
        self.dataset.time_series_depth = None
        return self.dataset


class RocketRepresentation():

    def __init__(self, config: Configuration, dataset: Dataset):
        self.config = config
        self.dataset = dataset

    def create_representation(self):
        x_train, x_val, x_test = self.dataset.x_train, self.dataset.x_test_val, self.dataset.x_test

        # Cast is necessary because rocket seems to expect 64 bit values
        x_train_casted = x_train.astype('float64')
        x_val_casted = x_val.astype('float64')
        x_test_casted = x_test.astype('float64')

        print('Transforming from array to dataframe...')
        x_train_df = self.array_to_ts_df(x_train_casted)
        x_val_df = self.array_to_ts_df(x_val_casted)
        x_test_df = self.array_to_ts_df(x_test_casted)

        rocket = Rocket(num_kernels=self.config.rocket_kernels,
                        normalise=False, random_state=self.config.random_seed)

        print('Fitting to train dataset...')
        rocket.fit(x_train_df)

        print('Transforming train dataset...')
        x_train = rocket.transform(x_train_df).values
        print('Transforming validation dataset...')
        x_val = rocket.transform(x_val_df).values
        print('Transforming test dataset...')
        x_test = rocket.transform(x_test_df).values

        # Convert back into datatype that is equivalent to the one used by the neural net
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        np.save(self.dataset.dataset_folder + self.config.rocket_features_train, x_train)
        np.save(self.dataset.dataset_folder + self.config.rocket_features_val, x_val)
        np.save(self.dataset.dataset_folder + self.config.rocket_features_test, x_test)

        print('Representation generation finished.')
        print('Resulting shapes of train, val and test: ', x_train.shape, x_val.shape, x_test.shape)

    # Numpy dataset must be converted to expected format described
    # @ https://www.sktime.org/en/latest/examples/loading_data.html
    def array_to_ts_df(self, array):
        # Input : (Example, Timestamp, Feature)
        # Temp 1: (Example, Feature, Timestamp)
        array_transformed = np.einsum('abc->acb', array)

        # No simpler / more elegant solution via numpy or pandas found
        # Create list of examples with list of features containing a pandas series of  timestamp values
        # Temp 2: (Example, Feature, Series of timestamp values)
        list_of_examples = []

        for example in array_transformed:
            ex = []
            for feature in example:
                ex.append(pd.Series(feature))

            list_of_examples.append(ex)

        # Conversion to dataframe with expected format
        return pd.DataFrame(data=list_of_examples)

    def load_into_dataset(self):
        self.dataset.x_train = np.load(self.dataset.dataset_folder + self.config.rocket_features_train)
        self.dataset.x_test_val = np.load(self.dataset.dataset_folder + self.config.rocket_features_val)
        self.dataset.x_test = np.load(self.dataset.dataset_folder + self.config.rocket_features_test)

        self.dataset.time_series_length = None
        self.dataset.time_series_depth = None
        return self.dataset


if __name__ == '__main__':
    config = Configuration()

    p = psutil.Process()
    cores = p.cpu_affinity()
    p.cpu_affinity(cores[0:config.max_parallel_cores])

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    # The representation that should be created must be switched manually here
    rep = TSFreshRepresentation(config, dataset)
    rep.create_representation()
