import sys

import tensorflow as tf
from spektral.layers import GATConv
from tensorflow.keras import layers

from configuration.Hyperparameter import Hyperparameters


class Convolutions(layers.Layer):

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper

        if len(self.hyper.conv_filters) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible. ')
            sys.exit(-1)

        if self.hyper.conv_filters[-1] != 1:
            print('Last convolution layer must have 1 as number of filters in order to match '
                  'the expected input shape in subsequent layers. ')
            sys.exit(-1)

        self.conv_layers = []

        layer_properties = list(zip(self.hyper.conv_filters, self.hyper.conv_kernel_size, self.hyper.conv_strides))

        for units, kernel_size, strides in layer_properties:
            print('Adding feature wise convolutions with {} filters per feature, '
                  '{} kernels and {} strides ...'.format(units, kernel_size, strides))

            # Based on https://stackoverflow.com/a/64990902
            conv_layer = tf.keras.layers.Conv1D(
                filters=units * self.hyper.time_series_depth,  # Configured filter number for each feature
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.keras.activations.relu,
                padding='causal',  # Recommended for temporal data, see https://bit.ly/3fvY1Qu
                groups=self.hyper.time_series_depth,  # Treat each feature as a separate input
                data_format='channels_last')
            self.conv_layers.append(conv_layer)

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
        return x


class FeatureGAT():

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper

        print('Adding feature based graph attention layer ...')
        self.feature_graph_layer = GATConv(channels=self.hyper.time_series_length)

    def create_model(self):
        # Main input has shape (F,) with F = Number of node features,
        # what corresponds to the number of timestamps for the feature oriented GAT layer
        X_input = tf.keras.Input(shape=(self.hyper.time_series_length,))

        # Adjacency matrix input has shape (N,) with N = Number of nodes in the graph,
        # what corresponds to the features for the feature oriented GAT layer
        A_input = tf.keras.Input(shape=(self.hyper.time_series_depth,), sparse=True)

        output = self.feature_graph_layer([X_input, A_input])
        return tf.keras.Model([X_input, A_input], output)


class TimeGAT():

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper

        print('Adding time based graph attention layer ...')
        self.time_graph_layer = GATConv(channels=self.hyper.time_series_depth)

    def create_model(self):
        # Main input has shape (F,) with F = Number of node features,
        # what corresponds to the number of features for the time oriented GAT layer
        X_input = tf.keras.Input(shape=(self.hyper.time_series_depth,))

        # Adjacency matrix input has shape (N,) with N = Number of nodes in the graph,
        # what corresponds to the number of timestamps for the time oriented GAT layer
        A_input = tf.keras.Input(shape=(self.hyper.time_series_length,), sparse=True)

        output = self.time_graph_layer([X_input, A_input])
        return tf.keras.Model([X_input, A_input], output)


class GRU(layers.Layer):

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper
        self.gru_layers = []

        for units in self.hyper.d1_gru_units:
            print('Adding GRU layer with {} units ...'.format(units))
            self.gru_layers.append(tf.keras.layers.GRU(units=units, return_sequences=True))

        # In case the the gru output is reconstructed by the vae, it's output shape must match the time series
        # to be able to use it in score calculation.
        # So if the unit size in the last gru layer does not match the expected value, i. e. the time series depth
        # we need to add an additional layer
        if 'reconstruct_gru' in self.hyper.variants and self.hyper.d1_gru_units[-1] != self.hyper.time_series_depth:
            print('Adding additional GRU layer with {} units for output shape compatibility ...'.format(
                self.hyper.time_series_depth))
            self.gru_layers.append(tf.keras.layers.GRU(units=self.hyper.time_series_depth, return_sequences=True))

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for gru_layer in self.gru_layers:
            x = gru_layer(x)

        return x


class ForecastingModel(layers.Layer):

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper

        self.fc_layers = []

        for units in self.hyper.d2_fc_units[0:-1]:
            print('Adding dense layer to forecasting model with {} units and ReLu activation ...'.format(units))
            self.fc_layers.append(tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu))

        # If the configured number of units in the last layer matches the required number, i. e. the time series depth,
        # we equip this one with a sigmoid activation and use it as output layer.
        # If it does not, it is added as hidden layer and an additional layer with the correct number of units
        # and sigmoid activation is added additionally.
        units_last_layer = self.hyper.d2_fc_units[-1]
        if units_last_layer == self.hyper.time_series_depth:
            print('Adding last dense layer to forecasting model with {} '
                  'units and sigmoid activation ...'.format(units_last_layer))
            self.fc_layers.append(tf.keras.layers.Dense(units=units_last_layer,
                                                        activation=tf.keras.activations.sigmoid))
        else:
            print('Adding dense layer to forecasting model with {} units and ReLu activation ...'.format(
                units_last_layer))
            self.fc_layers.append(tf.keras.layers.Dense(units=units_last_layer, activation=tf.keras.activations.relu))

            print('Adding additional dense layer with {} units (and sigmoid activation) '
                  'for output shape compatibility ...'.format(self.hyper.time_series_depth))
            self.fc_layers.append(tf.keras.layers.Dense(units=self.hyper.time_series_depth,
                                                        activation=tf.keras.activations.sigmoid))

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for fc_layer in self.fc_layers:
            x = fc_layer(x, training=training)

        return x


class GRU_ForecastingModel(layers.Layer):

    def __init__(self, hyper: Hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyper: Hyperparameters = hyper

        self.layers = []

        for units in self.hyper.d2_fc_units:
            print('Adding GRU layer to forecasting model with {} units ...'.format(units))
            self.layers.append(tf.keras.layers.GRU(units=units, return_sequences=True))

        self.layers.append(
            tf.keras.layers.Dense(units=self.hyper.time_series_depth, activation=tf.keras.activations.sigmoid))

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        for layer in self.layers:
            x = layer(x, training=training)

        return x
