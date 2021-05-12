import numpy as np


class PreSplitCleaner:

    def __init__(self, config, examples, labels, next_values, failure_times, window_times):
        self.config = config
        self.examples = examples
        self.labels = labels
        self.next_values = next_values
        self.failure_times = failure_times
        self.window_times = window_times

    def clean(self):
        # Must be executed in this order to ensure labels in data matches the ones defined in cleaning_info
        self.rename_labels()
        self.remove_examples_by_label()

    def rename_labels(self):
        for old, new in self.config.label_renaming.items():
            self.labels = np.char.replace(self.labels, old, new)

        for old, new in self.config.label_renaming_overall.items():
            self.labels = np.char.replace(self.labels, old, new)

        for old, new in self.config.label_renaming_overall.items():
            self.config.unused_labels = np.char.replace(self.config.unused_labels, old, new)

    def remove_examples_by_label(self):
        print()
        for label in self.config.unused_labels:
            x = np.argwhere(self.labels == label)
            print('Number of examples removed with label ', label, ': \t', len(x))
        print()

        remaining_examples_indices = np.isin(self.labels, self.config.unused_labels, invert=True)
        self.apply_indices_to_all(remaining_examples_indices)

    def apply_indices_to_all(self, indices_to_reduce_to):
        self.examples = self.examples[indices_to_reduce_to]
        self.labels = self.labels[indices_to_reduce_to]
        self.next_values = self.next_values[indices_to_reduce_to]
        self.failure_times = self.failure_times[indices_to_reduce_to]
        self.window_times = self.window_times[indices_to_reduce_to]

    def return_all(self):
        return self.examples, self.labels, self.next_values, self.failure_times, self.window_times


class PostSplitCleaner:

    def __init__(self, config,
                 x_train, x_test,
                 y_train, y_test,
                 next_values_train, next_values_test,
                 failure_times_train, failure_times_test,
                 window_times_train, window_times_test):
        self.config = config

        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        self.next_values_train, self.next_values_test = next_values_train, next_values_test
        self.failure_times_train, self.failure_times_test = failure_times_train, failure_times_test
        self.window_times_train, self.window_times_test = window_times_train, window_times_test

    def clean(self):
        self.move_from_train_to_test()

    def move_from_train_to_test(self):
        for failure_time in self.config.transfer_from_train_to_test:
            x = np.argwhere(self.failure_times_train == failure_time)
            print('Number of examples moved from train to test for failure time ', failure_time, ': \t', len(x))

        indices_to_extract = np.isin(self.failure_times_train, self.config.transfer_from_train_to_test)

        examples_ex = self.x_train.copy()[indices_to_extract]
        labels_ex = self.y_train.copy()[indices_to_extract]
        next_values_ex = self.next_values_train.copy()[indices_to_extract]
        failure_times_ex = self.failure_times_train.copy()[indices_to_extract]
        window_times_ex = self.window_times_train.copy()[indices_to_extract]

        self.x_test = np.concatenate([self.x_test, examples_ex])
        self.y_test = np.concatenate([self.y_test, labels_ex])
        self.next_values_test = np.concatenate([self.next_values_test, next_values_ex])
        self.failure_times_test = np.concatenate([self.failure_times_test, failure_times_ex])
        self.window_times_test = np.concatenate([self.window_times_test, window_times_ex])

        indices_kept = np.isin(self.failure_times_train, self.config.transfer_from_train_to_test, invert=True)
        self.x_train = self.x_train[indices_kept]
        self.y_train = self.y_train[indices_kept]
        self.next_values_train = self.next_values_train[indices_kept]
        self.failure_times_train = self.failure_times_train[indices_kept]
        self.window_times_train = self.window_times_train[indices_kept]

    def return_all(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.next_values_train, \
               self.next_values_test, self.failure_times_train, self.failure_times_test, \
               self.window_times_train, self.window_times_test
