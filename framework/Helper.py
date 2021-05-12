import os
import shutil
from datetime import datetime
from os import listdir

import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from framework.Dataset import Dataset


class FrameworkTrainer:

    def __init__(self, config, hyper, dataset, model):
        self.config: Configuration = config
        self.hyper: Hyperparameters = hyper
        self.dataset: Dataset = dataset
        self.model: tf.keras.Model = model

        if self.hyper.gradient_cap_enabled:
            self.optimizer = tf.optimizers.Adam(self.hyper.learning_rate, clipnorm=self.hyper.gradient_cap)
        else:
            self.optimizer = tf.optimizers.Adam(self.hyper.learning_rate)

        # Initialise check point manger which stores the weights after each epoch to be able to retrieve them
        # after selection by early stopping

        model_name_date_part = datetime.now().strftime("%m-%d-%H-%M-%S/")
        prefix = self.config.hyper_file.split('/')[-1].split('.')[0]
        self.selected_model_name = prefix + '_' + model_name_date_part
        self.selected_model_path = self.config.models_folder + self.selected_model_name
        self.training_model_path = self.config.models_folder + 'temp_' + model_name_date_part
        self.checkpoint_path = self.training_model_path + 'weights-{epoch:d}'

    def clean_up(self, early_stopping, current_epoch):
        """
        Deletes old temporarily saved weights.

        :param early_stopping:
        :param current_epoch:
        :return:
        """

        early_stopping: EarlyStopping = early_stopping

        if early_stopping.enable_stopping:
            lower_limit = early_stopping.best_loss_index - 1
        else:
            lower_limit = current_epoch - self.config.model_files_stored - 1

        for file in listdir(self.training_model_path):

            try:
                epoch_of_file = int(file.split('.')[0].split('-')[-1])
                if epoch_of_file <= lower_limit:
                    os.remove(self.training_model_path + file)
            except ValueError:
                pass
            except Exception as e:
                print(e)

    def train_model(self):
        """
        Method for execution the model training process
        """
        early_stopping = EarlyStopping(self, self.hyper.early_stopping_enabled, self.hyper.early_stopping_limit)
        loss_history_train = []
        loss_metric_train = tf.keras.metrics.Mean()

        x_train, next_values_train = self.dataset.create_batches(self.hyper.batch_size,
                                                                 [self.dataset.x_train,
                                                                  self.dataset.next_values_train])

        x_train_val, next_values_train_val = self.dataset.create_batches(self.hyper.batch_size,
                                                                         [self.dataset.x_train_val,
                                                                          self.dataset.next_values_train_val])

        for epoch in range(self.hyper.epochs):
            print("Epoch %d" % (epoch,))

            for step, (x_batch_train, next_values_batch_train) in enumerate(zip(x_train, next_values_train)):
                self.train_step(x_batch_train, next_values_batch_train, loss_metric_train)

                if step % 50 == 0:
                    print("\tStep %d: mean loss = %.4f" % (step, loss_metric_train.result()))

            loss_train_batch = loss_metric_train.result()
            loss_history_train.append(loss_train_batch)
            loss_metric_train.reset_states()

            self.model.save_weights(self.checkpoint_path.format(epoch=epoch))

            # Check early stopping criterion --> Has the loss on the validation set not decreased?
            best_epoch = early_stopping.execute(epoch, x_train_val, next_values_train_val)
            self.clean_up(early_stopping, epoch)

            if best_epoch > 0:
                print('Model from epoch %d was selected by early stopping.' % best_epoch)
                print('Training process will be stopped now.')

                self.save_model(best_epoch)

                return

        self.save_model(epoch=self.hyper.epochs - 1)

    def train_step(self, batch, y_next_true, loss_metric):
        """
        Executes a single training step.

        :param batch: The batch of training data that should be used for this step
        :param y_next_true: The ground truth labels for the examples in batch
        :param loss_metric: The metric object the loss for this step should be stored in.
        """

        with tf.GradientTape() as tape:
            loss = self.compute_loss(batch, y_next_true)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            loss_metric(loss)

    def compute_loss(self, batch, y_next_true):
        """
        Computes the current loss of the model

        :param batch: Batch for which the loss should be computed
        :param y_next_true: The true values for timestamp t+1 for each example with timestamps [0,t]
        :return: The computed loss
        """

        # Get the output of the gru layer for the input which serves as input to the reconstruction + forecasting model
        gru_output = self.model(batch, training=True)

        # Forecasting model loss calculation
        # Using mse yields the same result as RMSE and is more stable
        y_next_pred = self.model.forecasting_model(gru_output, training=True)
        y_next_pred = y_next_pred[:, -1, :]  # only get the prediction for the last timestamp

        mse_for = tf.keras.losses.MeanSquaredError()
        loss_for = mse_for(y_next_true, y_next_pred)

        # Reconstruction model loss calculation
        # Like VAE based on: https://bit.ly/3oRMiQz
        mse_rec = tf.keras.losses.MeanSquaredError()
        reconstructed_output = self.model.reconstruction_model(gru_output)
        reconstruction_target = gru_output if 'reconstruct_gru' in self.hyper.variants else batch

        loss_rec = mse_rec(reconstruction_target, reconstructed_output)
        loss_rec += sum(self.model.reconstruction_model.losses)  # Add KLD regularization loss

        # Overall loss
        loss = loss_for + loss_rec

        return loss

    def save_model(self, epoch):
        """
        Saves the model associated with the epoch to a separate directory

        :param epoch: Epoch of the selected model (based on early stopping or maximum number of epochs)
        """

        # Reload weights form the checkpoint with the best loss
        self.model.load_weights(self.checkpoint_path.format(epoch=epoch))

        # Due to the model being created via subclassing we can't save the complete model, only the weights
        self.model.save_weights(self.selected_model_path + 'weights')

        # For this reason we need to store the hyperparameters, so we can recreate the same model before loading weights
        self.hyper.write_to_file(self.selected_model_path + "hyperparameters_used.json")

        # Delete the folder with the temporary models if enabled
        if self.config.delete_temp_models_after_training:
            shutil.rmtree(self.training_model_path, ignore_errors=False)
            print('Deleted temporary files in {}'.format(self.training_model_path))

        # Store the name of the selected model in the current config object
        # so a direct test after training (if TrainTest.py is used) loads this model
        self.config.change_current_model(self.selected_model_name)

        print('Location of saved model:', self.selected_model_path, '\n')


class EarlyStopping:

    def __init__(self, framework, enable_stopping: bool, early_stopping_limit: int):
        """
        Evaluates the current model on the validation set and keeps track of the loss

        :param framework: The framework instance whose model should be used
        :param enable_stopping: Whether the validation result actually stops the training when the limit is reached
        :param early_stopping_limit: The limit of epochs, after which the training is stopped if the loss didn't improve
        """

        self.framework = framework
        self.enable_stopping = enable_stopping
        self.early_stopping_limit = early_stopping_limit
        self.best_loss_index = 0
        self.no_improvement_counter = 0
        self.loss_history_val = []

    def execute(self, epoch, x_val, next_values_val):
        """
        Should be called after each epoch to execute the loss calculation on the validation set

        :param epoch: The epoch of the training the evaluation is executed for
        :param x_val: Validation data set
        :param next_values_val: Values of the next timestamp for the examples in the validation set
        :return: -1 if the training process shouldn't be stopped or the selected epoch (as a positive integer)
        """
        loss_metric_val = tf.keras.metrics.Mean()

        for step, (x_batch_val, next_values_batch_val) in enumerate(zip(x_val, next_values_val)):
            loss = self.framework.compute_loss(x_batch_val, next_values_batch_val)
            loss_metric_val(loss)

        loss_val_batch = loss_metric_val.result().numpy()
        self.loss_history_val.append(loss_val_batch)

        print("\n\tLoss on validation set after epoch %d: %.4f\n" % (epoch, float(loss_val_batch),))

        if not np.isfinite(loss_val_batch):
            print('Stopping training because of NaN values.')

            if self.enable_stopping:
                return self.best_loss_index
            else:
                return epoch - 1

        # Check if the loss of the last epoch is better than the best loss
        # If so reset the early stopping progress else continue approaching the limit
        if loss_val_batch < self.loss_history_val[self.best_loss_index]:
            self.no_improvement_counter = 0
            self.best_loss_index = epoch
        else:
            self.no_improvement_counter += 1

        # Check if the limit was reached and if stopping is enabled
        if self.enable_stopping and self.no_improvement_counter >= self.early_stopping_limit:
            return self.best_loss_index
        else:
            return -1


class Helper:

    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters

    def get_feature_oriented_matrix(self):
        """
        The adjacency matrix used by the feature oriented gat layer has 1.0 at each position
        because this gat layer is modeled as a complete graph

        :return: np.ndarray with shape features x features and all positions equal to 1
            except pairs (i,i) (see https://en.wikipedia.org/wiki/Adjacency_matrix#Trivial_graphs)
        """
        nbr_features = self.hyperparameters.time_series_depth
        matrix = np.ones(shape=(nbr_features, nbr_features), dtype=np.float)
        np.fill_diagonal(matrix, val=0)
        return matrix

    def get_time_oriented_matrix(self, sliding_window_size):
        """
        The adjacency matrix used by the time oriented gat layer

        :param sliding_window_size: The size of the sliding window,
            i. e. the number consecutive timestamps connected in the graph. Must be an odd number.
        :return: np.ndarray with shape features x features. Values: 0/1, based on sliding window
        """
        nbr_timestamps = self.hyperparameters.time_series_length

        if sliding_window_size < 3 or sliding_window_size > nbr_timestamps:
            print('Unsuitable sliding window size configured:', sliding_window_size)
            print('Using complete linkage instead.\n')
            matrix = np.ones(shape=(nbr_timestamps, nbr_timestamps), dtype=np.float)
            np.fill_diagonal(matrix, val=0)
            return matrix

        assert sliding_window_size % 2 != 0, 'Sliding window size should be an odd number.'
        sliding_window_half = sliding_window_size // 2
        matrix = np.zeros(shape=(nbr_timestamps, nbr_timestamps), dtype=np.float)

        # use each timestamp i as a central point and set floor(sliding_window_size/2) many
        # timestamps left and right of it to 1, but not i itself
        for i in range(nbr_timestamps):
            lower = max(0, i - sliding_window_half)
            upper = min(nbr_timestamps, i + sliding_window_half + 1)

            index_interval = [j for j in range(lower, upper) if i != j]
            matrix[i, index_interval] = 1

        return matrix
