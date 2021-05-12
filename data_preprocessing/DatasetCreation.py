import gc
import os
import pickle
import sys
import threading

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from data_preprocessing.DatasetCleaning import PostSplitCleaner, PreSplitCleaner
from configuration.Enums import TrainTestSplitMode
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration


class CaseSplitter(threading.Thread):

    def __init__(self, case_info, df: pd.DataFrame):
        super().__init__()
        self.case_label = case_info[0]
        self.start_timestamp_case = case_info[1]
        self.end_timestamp_case = case_info[2]
        self.failure_time = case_info[3]
        self.df = df
        self.result = None

    def run(self):
        try:
            # reassignment is necessary
            case_label = self.case_label
            failure_time = self.failure_time
            start_timestamp_case = self.start_timestamp_case
            end_timestamp_case = self.end_timestamp_case
            df = self.df

            short_label = case_label[0:25] + '...' if len(case_label) > 25 else case_label
            case_info = ['Processing ' + short_label, str(start_timestamp_case), str(end_timestamp_case),
                         'Failure time:', str(failure_time)]

            print("\t{: <50} {: <30} {: <30} {: <20} {: <25}".format(*case_info))

            # basic checks for correct timestamps
            if end_timestamp_case < start_timestamp_case:
                raise KeyError()
            if start_timestamp_case < df.first_valid_index():
                start_timestamp_case = df.first_valid_index()
            if end_timestamp_case > df.last_valid_index():
                end_timestamp_case = df.last_valid_index()

            # extract the part of the case from the dataframe
            self.result = df[start_timestamp_case: end_timestamp_case]

        except KeyError:
            print('CAUTION: Unknown timestamp or wrong order of start/end in at least one case')


# split the dataframe into the failure cases
def split_by_cases(df: pd.DataFrame, data_set_counter, config: Configuration):
    print('\nSplit data by cases with the configured timestamps')

    # get the cases of the dataset after which it should be split
    cases_info = config.cases_datasets[data_set_counter]
    # print(cases_info[1])
    cases = []  # contains dataframes from sensor data
    labels = []  # contains the label of the dataframe
    failures = []  # contains the associated failure time stamp
    threads = []

    # prepare case splitting threads
    for i in range(len(cases_info)):
        t = CaseSplitter(cases_info[i], df)
        threads.append(t)

    # execute threads with the configured amount of parallel threads
    thread_limit = config.max_parallel_cores if len(threads) > config.max_parallel_cores else len(threads)
    threads_finished = 0

    while threads_finished < len(threads):
        if threads_finished + thread_limit > len(threads):
            thread_limit = len(threads) - threads_finished

        r = threads_finished + thread_limit

        print('Processing case', threads_finished, 'to', r - 1)

        for i in range(threads_finished, r):
            threads[i].start()

        for i in range(threads_finished, r):
            threads[i].join()

        for i in range(threads_finished, r):
            if threads[i].result is not None:
                cases.append(threads[i].result)
                labels.append(threads[i].case_label)
                failures.append(threads[i].failure_time)

        threads_finished += thread_limit

    return cases, labels, failures


def extract_single_example(df: pd.DataFrame):
    # No reduction is used if overlapping window is applied
    # because data is down sampled before according parameter sampling frequency
    sampled_values = df.to_numpy()

    # Split sampled values into actual example and values of next timestamp
    example = sampled_values[0:-1]
    next_values = sampled_values[-1]

    # -2 instead of last index because like the version without overlapping time window
    # the last value is not part of the actual example
    time_window_pattern = "%Y-%m-%d %H:%M:%S"
    time_window_string = df.index[0].strftime(time_window_pattern), df.index[-2].strftime(time_window_pattern)

    # print('Example:', example.shape, df.index[0], df.index[0:-1][-1])
    # print('\tNext:', next_values.shape, df.index[-1])

    return example, next_values, time_window_string


def split_into_examples(df: pd.DataFrame, label: str, examples: [np.ndarray], labels_of_examples: [str],
                        config: Configuration,
                        failure_times_of_examples: [str], failure_time,
                        window_times_of_examples: [str], y, i_dataset, next_values: [np.ndarray]):
    start_time = df.index[0]
    end_time = df.index[-1]

    # slide over data frame and extract windows until the window would exceed the last time step
    while start_time + pd.to_timedelta(config.overlapping_window_step_seconds, unit='s') < end_time:

        # generate a list with indexes for window
        # time_series_length +1 because split the result into actual examples and values of next timestamp
        overlapping_window_indices = pd.date_range(start_time, periods=config.time_series_length + 1,
                                                   freq=config.resample_frequency)

        example, next_values_example, time_window_string = extract_single_example(df.asof(overlapping_window_indices))

        # store information for each example calculated by the threads
        labels_of_examples.append(label)
        examples.append(example)
        next_values.append(next_values_example)
        window_times_of_examples.append(time_window_string)

        # store failure time or special string if no failure example
        if label == 'no_failure':
            failure_times_of_examples.append("noFailure-" + str(i_dataset) + "-" + str(y))
        else:
            failure_times_of_examples.append(str(failure_time))

        # update next start time for next window
        start_time = start_time + pd.to_timedelta(config.overlapping_window_step_seconds, unit='s')


def normalise(x_train: np.ndarray, x_test: np.ndarray, config: Configuration, next_values_train, next_values_test):
    print('\nExecute normalisation')
    length = x_train.shape[2]

    for i in range(length):
        scaler = MinMaxScaler(feature_range=(0, 1))

        # reshape column vector over each example and timestamp to a flatt array
        # necessary for normalisation to work properly
        shape_before = x_train[:, :, i].shape
        x_train_shaped = x_train[:, :, i].reshape(shape_before[0] * shape_before[1], 1)

        # learn scaler only on training data (best practice)
        x_train_shaped = scaler.fit_transform(x_train_shaped)

        # reshape back to original shape and assign normalised values
        x_train[:, :, i] = x_train_shaped.reshape(shape_before)

        # normalise test data
        shape_before = x_test[:, :, i].shape
        x_test_shaped = x_test[:, :, i].reshape(shape_before[0] * shape_before[1], 1)
        x_test_shaped = scaler.transform(x_test_shaped)
        x_test[:, :, i] = x_test_shaped.reshape(shape_before)

        # normalise next values
        shape_before = next_values_train[:, i].shape
        next_train_shaped = next_values_train[:, i].reshape(shape_before[0], 1)
        next_train_shaped = scaler.transform(next_train_shaped)
        next_values_train[:, i] = next_train_shaped.reshape(shape_before)

        shape_before = next_values_test[:, i].shape
        next_test_shaped = next_values_test[:, i].reshape(shape_before[0], 1)
        next_test_shaped = scaler.transform(next_test_shaped)
        next_values_test[:, i] = next_test_shaped.reshape(shape_before)

        # export scaler to use with live data
        scaler_filename = config.scaler_folder + 'scaler_' + str(i) + '.save'
        joblib.dump(scaler, scaler_filename)

    return x_train, x_test, next_values_train, next_values_test


def determine_train_test_indices(config: Configuration, examples_array, labels_array, failure_times_array):
    if config.split_mode == TrainTestSplitMode.ENSURE_NO_MIX:

        # Split into train and test considering that examples from a single failure run don't end up in both
        # print('\nExecute train/test split in ENSURE_NO_MIX mode.')
        enc = OrdinalEncoder()
        enc.fit(failure_times_array.reshape(-1, 1))
        failure_times_array_groups = enc.transform(failure_times_array.reshape(-1, 1))

        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_split_size, random_state=config.random_seed)

        train_indices, test_indices = list(gss.split(examples_array, labels_array, failure_times_array_groups))[0]

    elif config.split_mode == TrainTestSplitMode.ANOMALY_DETECTION:

        # This means all failure examples are in test
        # Only no_failure examples will be split based on configured percentage
        print('\nExecute train/test split in ANOMALY_DETECTION mode.')

        # Split examples into normal and failure cases
        failure_indices = np.argwhere(labels_array != 'no_failure').flatten()
        no_failure_indices = np.argwhere(labels_array == 'no_failure').flatten()

        # Execute recording instance based splitting only for no_failures
        # For which the input arrays are first of all reduced to those examples
        nf_examples = examples_array[no_failure_indices]
        nf_labels = labels_array[no_failure_indices]
        nf_failure_times = failure_times_array[no_failure_indices]

        enc = OrdinalEncoder()
        enc.fit(nf_failure_times.reshape(-1, 1))
        nf_groups = enc.transform(nf_failure_times.reshape(-1, 1))

        # Split the no failure only examples based on the recording instances and the split size
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_split_size, random_state=config.random_seed)
        nf_train_indices_in_reduced, nf_test_indices_in_reduced = \
            list(gss.split(nf_examples, nf_labels, nf_groups))[0]

        # Trace back the indices of the reduced arrays to the indices of the complete arrays
        nf_train_indices = no_failure_indices[nf_train_indices_in_reduced]
        nf_test_indices = no_failure_indices[nf_test_indices_in_reduced]

        # Combine indices to full lists
        # Train part only consists of the  train part of the no failure split,
        # whereas the test part consists of the test part of the no failure split as well as failure examples
        train_indices = list(nf_train_indices)
        test_indices = list(failure_indices) + list(nf_test_indices)
    else:
        raise ValueError()

    return train_indices, test_indices


def main():
    config = Configuration()  # Get config for data directory

    checker = ConfigChecker(config, None, 'preprocessing', training=None)
    checker.pre_init_checks()

    config.import_timestamps()
    number_data_sets = len(config.datasets)

    # list of all examples
    examples: [np.ndarray] = []
    labels_of_examples: [str] = []
    next_values: [np.ndarray] = []
    failure_times_of_examples: [str] = []
    window_times_of_examples: [str] = []

    attributes = None

    for i in range(number_data_sets):
        print('\n\nImporting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')

        # read the imported dataframe from the saved file
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned

        with open(path_to_file, 'rb') as f:
            df: pd.DataFrame = pickle.load(f)

        # split the dataframe into the configured cases
        cases_df, labels_df, failures_df = split_by_cases(df, i, config)

        if i == 0:
            attributes = np.stack(df.columns, axis=0)

        del df
        gc.collect()

        # split the case into examples, which are added to the list of of all examples
        number_cases = len(cases_df)

        print()
        for y in range(number_cases):
            df = cases_df[y]

            if len(df) <= 0:
                print(i, y, ' is empty!')
                continue

            start = df.index[0]
            end = df.index[-1]
            secs = (end - start).total_seconds()
            print('Splitting case', y, '/', number_cases - 1, 'into examples. Length:', secs, " Start: ", start,
                  " End: ", end)
            split_into_examples(df, labels_df[y], examples, labels_of_examples, config,
                                failure_times_of_examples, failures_df[y],
                                window_times_of_examples, y, i, next_values)
        print()
        del cases_df, labels_df, failures_df
        gc.collect()

    # convert lists of arrays to numpy array
    examples_array = np.stack(examples, axis=0)
    labels_array = np.stack(labels_of_examples, axis=0)
    next_values_array = np.stack(next_values, axis=0)
    failure_times_array = np.stack(failure_times_of_examples, axis=0)
    window_times_array = np.stack(window_times_of_examples, axis=0)

    del examples, labels_of_examples, failure_times_of_examples, window_times_of_examples
    gc.collect()

    cleaner = PreSplitCleaner(config, examples_array, labels_array, next_values_array, failure_times_array,
                              window_times_array)

    print('\nExamples before pre train/test split cleaning:', examples_array.shape[0])
    cleaner.clean()
    examples_array, labels_array, next_values_array, failure_times_array, window_times_array = cleaner.return_all()
    print('Examples after pre train/test split cleaning:', examples_array.shape[0])

    train_indices, test_indices = determine_train_test_indices(config, examples_array, labels_array,
                                                               failure_times_array)

    x_train, x_test = examples_array[train_indices], examples_array[test_indices]
    y_train, y_test = labels_array[train_indices], labels_array[test_indices]
    next_values_train, next_values_test = next_values_array[train_indices], next_values_array[test_indices]
    failure_times_train, failure_times_test = failure_times_array[train_indices], failure_times_array[test_indices]
    window_times_train, window_times_test = window_times_array[train_indices], window_times_array[test_indices]

    del examples_array, labels_array, next_values_array, failure_times_array, window_times_array
    gc.collect()

    # Execute some manual corrections
    cleaner = PostSplitCleaner(config,
                               x_train, x_test,
                               y_train, y_test,
                               next_values_train, next_values_test,
                               failure_times_train, failure_times_test,
                               window_times_train, window_times_test)

    print('\nExamples in train before:', x_train.shape[0])
    print('Examples in test before:', x_test.shape[0], '\n')

    cleaner.clean()

    x_train, x_test, y_train, y_test, next_values_train, next_values_test, \
    failure_times_train, failure_times_test, window_times_train, window_times_test = cleaner.return_all()

    print('\nExamples in train after:', x_train.shape[0])
    print('Examples in test after:', x_test.shape[0], '\n')

    # normalize each sensor stream to contain values in [0,1]
    x_train, x_test, next_values_train, next_values_test = normalise(x_train, x_test, config, next_values_train,
                                                                     next_values_test)

    # cast to float32 so it can directly be used as tensorflow input without casting
    x_train, x_test, = x_train.astype('float32'), x_test.astype('float32')
    next_values_train, next_values_test = next_values_train.astype('float32'), next_values_test.astype('float32')

    test_arrays = [x_test, y_test, next_values_test, window_times_test, failure_times_test]
    separated = separate_validation(test_arrays, config, config.test_val_split_size)

    if separated is not None:
        x_test, y_test, next_values_test, window_times_test, failure_times_test = separated[0]
        x_test_val, y_test_val, next_values_test_val, window_times_test_val, failure_times_test_val = separated[1]
        has_test_val = True
    else:
        x_test_val, y_test_val, next_values_test_val, window_times_test_val, failure_times_test_val = np.empty(
            [1]), np.empty([1]), np.empty(
            [1]), np.empty([1]), np.empty([1])
        has_test_val = False

    train_arrays = [x_train, y_train, next_values_train, window_times_train, failure_times_train]
    separated = separate_validation(train_arrays, config, config.train_val_split_size)

    if separated is not None:
        x_train, y_train, next_values_train, window_times_train, failure_times_train = separated[0]
        x_train_val, y_train_val, next_values_train_val, window_times_train_val, failure_times_train_val = separated[1]
        has_train_val = True
    else:
        x_train_val, y_train_val, next_values_train_val, window_times_train_val, failure_times_train_val = np.empty(
            [1]), np.empty([1]), np.empty(
            [1]), np.empty([1]), np.empty([1])
        has_train_val = False

    print("x_train:", x_train.shape, "x_test:", x_test.shape, "x_test_val:", x_test_val.shape, "x_train_val:",
          x_train_val.shape, )
    print("y_train:", y_train.shape, "y_test:", y_test.shape, "y_test_val:", y_test_val.shape, "y_train_val:",
          y_train_val.shape)

    print("next_values_train:", next_values_train.shape, "next_values_test:", next_values_test.shape,
          "next_values_test_val:", next_values_test_val.shape)
    print("failure_times_train:", failure_times_train.shape, "failure_times_test:", failure_times_test.shape,
          "failure_times_test_val:", failure_times_test_val.shape)
    print("window_times_train:", window_times_train.shape, "window_times_test:", window_times_test.shape,
          "window_times_test:", window_times_test_val.shape)
    print()
    print("Classes in the train set:\n", np.unique(y_train))
    print("Classes in the test set:\n", np.unique(y_test))
    print("Classes in the train validation set:\n", np.unique(y_train_val))
    print("Classes in the test validation set:\n", np.unique(y_test_val))

    # save the np arrays
    print('\nSaving as np arrays to ' + config.training_data_folder)

    np.save(config.training_data_folder + 'feature_names.npy', attributes)

    print('Saving train dataset ... ')
    np.save(config.training_data_folder + 'train_features.npy', x_train)
    np.save(config.training_data_folder + 'train_labels.npy', y_train)
    # Contains the associated time of a failure (if not no failure) for each example
    np.save(config.training_data_folder + 'train_failure_times.npy', failure_times_train)
    # Contains the start and end time stamp for each training example
    np.save(config.training_data_folder + 'train_window_times.npy', window_times_train)
    # Contain the values of the next timestamp after each training example
    np.save(config.training_data_folder + 'train_next_values.npy', next_values_train)

    print('Saving test dataset ... ')
    np.save(config.training_data_folder + 'test_features.npy', x_test)
    np.save(config.training_data_folder + 'test_labels.npy', y_test)
    np.save(config.training_data_folder + 'test_failure_times.npy', failure_times_test)
    np.save(config.training_data_folder + 'test_window_times.npy', window_times_test)
    np.save(config.training_data_folder + 'test_next_values.npy', next_values_test)

    if has_train_val:
        print('Saving training validation dataset ... ')
        np.save(config.training_data_folder + 'train_val_features.npy', x_train_val)
        np.save(config.training_data_folder + 'train_val_labels.npy', y_train_val)
        np.save(config.training_data_folder + 'train_val_failure_times.npy', failure_times_train_val)
        np.save(config.training_data_folder + 'train_val_window_times.npy', window_times_train_val)
        np.save(config.training_data_folder + 'train_val_next_values.npy', next_values_train_val)

    if has_test_val:
        print('Saving test validation dataset ... ')
        np.save(config.training_data_folder + 'test_val_features.npy', x_test_val)
        np.save(config.training_data_folder + 'test_val_labels.npy', y_test_val)
        np.save(config.training_data_folder + 'test_val_failure_times.npy', failure_times_test_val)
        np.save(config.training_data_folder + 'test_val_window_times.npy', window_times_test_val)
        np.save(config.training_data_folder + 'test_val_next_values.npy', next_values_test_val)


def separate_validation(test_arrays, config: Configuration, test_size):
    x_test_old, y_test_old, next_values_test_old, window_times_test_old, failure_times_test_old = test_arrays

    if test_size <= 0.0:
        print('No validation set seperated from the train set.')
        return None

    # test_size is the size of the validation set because the method expects to split train/test not test/validation
    test_indices, val_indices = train_test_split(np.arange(len(y_test_old)), test_size=test_size,
                                                 random_state=config.random_seed)

    window_times_val, failure_times_val = window_times_test_old[val_indices], failure_times_test_old[val_indices]
    x_val, y_val, next_values_val = x_test_old[val_indices], y_test_old[val_indices], next_values_test_old[val_indices]

    window_times_test, failure_times_test = window_times_test_old[test_indices], failure_times_test_old[test_indices]
    x_test, y_test, next_values_test = x_test_old[test_indices], y_test_old[test_indices], next_values_test_old[
        test_indices]

    val_arrays = [x_val, y_val, next_values_val, window_times_val, failure_times_val]
    reduced_test_arrays = [x_test, y_test, next_values_test, window_times_test, failure_times_test]
    return [reduced_test_arrays, val_arrays]


if __name__ == '__main__':
    main()
