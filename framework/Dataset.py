import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples, time steps, channels)
        self.y_train = None  # One hot encoded class labels (numExamples, numClasses)
        self.y_train_strings = None  # class labels as strings (numExamples, 1)
        self.y_train_anomaly_mask = None  # Boolean mask whether the example is an anomaly or not (numExamples, 1)
        self.num_train_instances = None
        self.next_values_train = None  # The attribute vector at time step n+1
        self.full_next_train = None  # The time series from time step 1 to n+1

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.y_test_anomaly_mask = None
        self.num_test_instances = None
        self.next_values_test = None
        self.full_next_test = None

        self.x_test_val = None
        self.y_test_val = None
        self.y_test_val_strings = None
        self.y_test_val_anomaly_mask = None
        self.num_test_val_instances = None
        self.next_values_test_val = None
        self.full_next_test_val = None

        self.x_train_val = None
        self.y_train_val = None
        self.y_train_val_strings = None
        self.y_train_val_anomaly_mask = None
        self.num_train_val_instances = None
        self.next_values_train_val = None
        self.full_next_train_val = None

        self.num_instances = None

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

        # total number of classes
        self.num_classes = None

        # np array that contains a list classes that occur in training OR test data set
        self.classes_total = None

        # np array that contains a list classes that occur in training AND test data set
        self.classes_in_both = None

        self.one_hot_index_to_string = {}

        # additional information for each example about their window time frame and failure occurrence time
        # self.window_times_train = None
        # self.window_times_test = None
        # self.window_times_val = None
        # self.failure_times_train = None
        # self.failure_times_test = None
        # self.failure_times_val = None

    def load_files(self):
        """
        Loads necessary files from disk
        """

        self.x_train = np.load(self.dataset_folder + 'train_features.npy')
        self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)

        self.x_test = np.load(self.dataset_folder + 'test_features.npy')
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)

        self.x_test_val = np.load(self.dataset_folder + 'test_val_features.npy')
        self.y_test_val_strings = np.expand_dims(np.load(self.dataset_folder + 'test_val_labels.npy'), axis=-1)

        self.x_train_val = np.load(self.dataset_folder + 'train_val_features.npy')
        self.y_train_val_strings = np.expand_dims(np.load(self.dataset_folder + 'train_val_labels.npy'), axis=-1)

        assert all(label == 'no_failure' for label in self.y_train_strings), 'Failure example in x_train'
        assert all(label == 'no_failure' for label in self.y_train_val_strings), 'Failure example in x_train_val'

        # names of the features (3. dim)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy', allow_pickle=True)

        self.next_values_train = np.load(self.dataset_folder + 'train_next_values.npy')
        self.next_values_test = np.load(self.dataset_folder + 'test_next_values.npy')
        self.next_values_test_val = np.load(self.dataset_folder + 'test_val_next_values.npy')
        self.next_values_train_val = np.load(self.dataset_folder + 'train_val_next_values.npy')

        # if not self.is_third_party_dataset:
        #     self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
        #     self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)
        #     self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
        #     self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)
        #     self.window_times_val = np.expand_dims(np.load(self.dataset_folder + 'val_window_times.npy'), axis=-1)
        #     self.failure_times_val = np.expand_dims(np.load(self.dataset_folder + 'val_failure_times.npy'), axis=-1)

    def load(self, print_info=True):
        """
        Load the dataset from the configured location
        :param print_info: Output of basic information about the dataset at the end
        """

        self.load_files()

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        self.one_hot_encoder = self.one_hot_encoder.fit(
            np.concatenate(
                (self.y_train_strings, self.y_test_strings, self.y_test_val_strings, self.y_train_val_strings), axis=0))

        # transforms the vector of labels into a one hot matrix
        self.y_train = self.one_hot_encoder.transform(self.y_train_strings)
        self.y_test = self.one_hot_encoder.transform(self.y_test_strings)
        self.y_test_val = self.one_hot_encoder.transform(self.y_test_val_strings)
        self.y_train_val = self.one_hot_encoder.transform(self.y_train_val_strings)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)
        self.y_test_strings = np.squeeze(self.y_test_strings)
        self.y_test_val_strings = np.squeeze(self.y_test_val_strings)
        self.y_train_val_strings = np.squeeze(self.y_train_val_strings)

        # create a boolean mask whether an example is an anomaly or not
        self.y_train_anomaly_mask = self.y_train_strings != 'no_failure'
        self.y_test_anomaly_mask = self.y_test_strings != 'no_failure'
        self.y_test_val_anomaly_mask = self.y_test_val_strings != 'no_failure'
        self.y_train_val_anomaly_mask = self.y_train_val_strings != 'no_failure'

        # Combine main example data and arrays with next values in such a way that the next_full array contains
        # the values t=1 ... t=n+1 (but not t=0) for a time series of length n
        next_temp = np.expand_dims(self.next_values_train, axis=1)
        self.full_next_train = np.concatenate([self.x_train, next_temp], axis=1)
        self.full_next_train = self.full_next_train[:, 1:, :]

        next_temp = np.expand_dims(self.next_values_test, axis=1)
        self.full_next_test = np.concatenate([self.x_test, next_temp], axis=1)
        self.full_next_test = self.full_next_test[:, 1:, :]

        next_temp = np.expand_dims(self.next_values_test_val, axis=1)
        self.full_next_test_val = np.concatenate([self.x_test_val, next_temp], axis=1)
        self.full_next_test_val = self.full_next_test_val[:, 1:, :]

        next_temp = np.expand_dims(self.next_values_train_val, axis=1)
        self.full_next_train_val = np.concatenate([self.x_train_val, next_temp], axis=1)
        self.full_next_train_val = self.full_next_train_val[:, 1:, :]

        ##
        # safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]
        self.num_test_instances = self.x_test.shape[0]
        self.num_test_val_instances = self.x_test_val.shape[0]
        self.num_train_val_instances = self.x_train_val.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances + self.num_test_val_instances + self.num_train_val_instances

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # get the unique classes and the corresponding number
        self.classes_total = np.unique(
            np.concatenate(
                (self.y_train_strings, self.y_test_strings, self.y_test_val_strings, self.y_train_val_strings), axis=0))
        self.num_classes = self.classes_total.size

        # Create two dictionaries to link/associate each class with all its training examples
        for integer_index, c in enumerate(self.classes_total):
            self.one_hot_index_to_string[integer_index] = c

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels

        if print_info:
            print()
            print('Dataset loaded:')
            print('Shape of training set (example, time, channels):', self.x_train.shape)
            print('Shape of test set (example, time, channels):', self.x_test.shape)
            print('Shape of train validation set (example, time, channels):', self.x_train_val.shape)
            print('Shape of test validation set (example, time, channels):', self.x_test_val.shape)
            print('Num of classes in all:', self.num_classes)
            # print('Classes used in training: ', len(self.y_train_strings_unique), " :", self.y_train_strings_unique)
            # print('Classes used in test: ', len(self.y_test_strings_unique), " :", self.y_test_strings_unique)
            # print('Classes in both: ', self.classes_in_both)
            print()

    def create_batches(self, batch_size, arrays: [np.ndarray]):
        """

        :param batch_size: The number of examples per batch
        :param arrays: A list of arrays that should be split into batches in the same way
            (maintaining the correct mappings by using the same seed)
        :return: A list of tensorflow datasets
        """

        processed_arrays = []

        # Convert into batches, use seed in order ensure same shuffling for next values so we dont change the "pairs"
        buffer_size, seed = 1024, self.config.random_seed

        for array in arrays:
            x = tf.data.Dataset.from_tensor_slices(array)
            x = x.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size, drop_remainder=False)
            processed_arrays.append(x)

        return processed_arrays
