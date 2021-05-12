import os

import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from execution.Evaluator import Evaluator
from framework.Dataset import Dataset
from framework.Helper import FrameworkTrainer, Helper
from framework.MTAD_GAT import MTAD_GAT
from framework.Ontology import SemanticModel
from framework.SymptomIdentification import SymptomIdentification


class Framework:

    def __init__(self, config: Configuration, dataset: Dataset, training):
        self.config: Configuration = config
        self.dataset: Dataset = dataset
        self.training = training
        self.hyper: Hyperparameters = Hyperparameters()
        self.model: tf.keras.Model

        self.load_hyperparameters()

        # Adjacency matrices for the GAT layers
        helper = Helper(self.hyper)
        feature_mask = helper.get_feature_oriented_matrix()
        time_mask = helper.get_time_oriented_matrix(self.hyper.sw_size_time_gat)

        self.model = MTAD_GAT(self.hyper, feature_mask, time_mask)

        if self.training:
            self.trainer = FrameworkTrainer(self.config, self.hyper, self.dataset, self.model)
        else:
            self.load_model()

        self.ontology = SemanticModel(self.config)
        self.ontology.import_from_file()

        self.reset_state()

    def reset_state(self):
        # Set seeds in order to get consistent results, mainly for sampling in the VAE when executing multiple run,
        # e.g. during grid search
        tf.random.set_seed(25011997)
        np.random.seed(25011997)

    def load_hyperparameters(self):
        """
        Loads hyperparameters. If training loads it from the configured file,
        if testing from the configured model directory
        """
        if self.training:
            file_path = self.config.hyper_file
            print('Creating model based on {} hyperparameter file'.format(self.config.hyper_file), '\n')

        else:
            file_path = self.config.directory_model_to_use + 'hyperparameters_used.json'

        self.hyper.load_from_file(file_path, self.config.use_hyper_file)
        self.hyper.set_time_series_properties(self.dataset)

    def load_model(self):
        """
        Load the configured model weights.
        """

        status = self.model.load_weights(self.config.directory_model_to_use + 'weights')
        status.expect_partial()

        if os.path.isfile(self.config.directory_model_to_use + self.config.ss_percentiles_train_file):
            self.ss_percentiles_train = np.load(
                self.config.directory_model_to_use + self.config.ss_percentiles_train_file)
        else:
            self.ss_percentiles_train = None

    def train_model(self):
        """
        Wrapper method for executing the training process directly on the framework object. Calls a framework trainer.
        """

        self.trainer.train_model()

        if self.config.percentile_calc_after_training:
            self.generate_percentiles_file(True)

    def executable_check(self):
        msg = 'Necessary hyperparameter is not set: {}'
        vars = [self.hyper.single_timestamp_anomaly_threshold, self.hyper.gamma,
                self.hyper.affected_timestamps_threshold]
        names = ['single_timestamp_anomaly_threshold', 'gamma', 'affected_timestamps_threshold']

        for name, var in zip(names, vars):
            assert var is not None, msg.format(name)

        if self.hyper.si_mode in ['train_percentile', 'jenks_percentile'] and self.ss_percentiles_train is None:
            raise ValueError('File containing SS percentiles of the training dataset was not loaded.'
                             'Can not use the configured SI mode')

    def test_model(self, print_results=False, generate=False):
        """

        :param print_results: Whether the results of this single model evaluation should be printed
        :param generate: If the percentiles file should be generated for the FPD.
            Can be used to disable this when this was already done by the grid search calling this function.
        :return:
        """

        self.executable_check()

        # Necessary for grid search of the relevance mapping
        self.ontology.update_scores(self.hyper.relevance_mapping)

        if print_results:
            # Output which model is used.
            set_name = 'validation' if self.config.test_with_validation_dataset else 'test'
            print('Evaluating model {} on the {} dataset. \n'.format(self.config.filename_model_to_use, set_name))
        else:
            self.config.enable_intermediate_output = False
            self.config.enable_config_output = False

        if generate:
            self.generate_percentiles_file()

        evaluator = Evaluator(self.config, self.dataset)

        # Create batches of relevant data objects
        if self.config.test_with_validation_dataset:
            arrays = [self.dataset.x_test_val, self.dataset.full_next_test_val,
                      np.arange(self.dataset.num_test_val_instances)]
        else:
            arrays = [self.dataset.x_test, self.dataset.full_next_test, np.arange(self.dataset.num_test_instances)]

        x, full_next, indices = self.dataset.create_batches(self.hyper.batch_size, arrays)

        for step, (x_batch, full_next, indices_batch) in enumerate(zip(x, full_next, indices)):
            batch_results = self.evaluate_batch(x_batch, full_next)

            anomaly_decisions = batch_results[0]
            a_t_vectors = batch_results[1]
            selected_symptoms = batch_results[2]
            st_result = batch_results[3]

            for index, ad, a_t, ss, st_r in zip(indices_batch, anomaly_decisions,
                                                a_t_vectors, selected_symptoms, st_result):
                intermediate_prediction = ad.numpy()
                st_components, st_scores = st_r

                if self.hyper.unaffected_component_threshold is None:
                    final_prediction = intermediate_prediction

                # In case of a detected anomaly check if it could be a false positive
                # by testing whether the score to the selected component with the highest probability
                # is  below a threshold
                elif intermediate_prediction:
                    final_prediction = True if st_scores[0] > self.hyper.unaffected_component_threshold else False
                    # Change the selected component to reflect this in the ST evaluation
                    if not final_prediction:
                        st_components = ['none' for _ in range(len(st_components))]
                else:
                    final_prediction = intermediate_prediction

                # replace returned component name with None str
                evaluator.add_ad_result(intermediate_prediction, a_t, index, final_prediction)
                evaluator.add_si_st_result(ss, st_components, st_scores, index)

        evaluator.calculate_results()

        if print_results:
            evaluator.print_results()

        if self.config.enable_config_output:
            print()
            print('Hyperparameter configuration used for this test:')
            print()
            self.hyper.print_hyperparameters()

        self.reset_state()
        return evaluator

    def evaluate_batch(self, x_batch, full_next):
        """
        Executes the hole evaluation process for a single batch, mainly for model testing

        :param x_batch: Array of examples that should be evaluated. Expected shape: [batch size, timestamps, features]
        :param full_next: Contains for each time step t in x_batch the values of time step t+1 at index t.
        :return: Whether each example is an anomaly, which timestamps are affected,
            the main symptoms and affected components
        """

        base_output = self.model([x_batch, full_next], training=False)

        are_anomaly, a_t_vectors = self.anomaly_decision(base_output)
        ss_vectors = self.symptom_score_calculation(base_output, a_t_vectors)

        selected_symptoms = self.batch_wrapper(method=self.symptom_identification, batch=ss_vectors)
        st_result = self.batch_wrapper(method=self.symptom_tracing, batch=ss_vectors)

        return are_anomaly, a_t_vectors, selected_symptoms, st_result

    def anomaly_decision(self, base_output: tf.Tensor):
        """
        Calculates which timestamps in an example are affected and if the overall example is classified as an anomaly

        :param base_output: The output of the MTAD_GAT model
        :return: Whether each example is classified as anomaly and which timestamps are affected.
        """

        nbr_affected_ts_threshold = self.hyper.affected_timestamps_threshold
        single_ts_anomaly_threshold = self.hyper.single_timestamp_anomaly_threshold

        # Formula 5
        reduced_to_timestamp = tf.reduce_mean(base_output, axis=2)

        # Return boolean vector which timestamps have a lager score than the defined threshold, Formula 6
        a_t_vector = tf.math.greater(reduced_to_timestamp, single_ts_anomaly_threshold)

        # Count how many timestamps are predicted to contain an anomaly, Formula 7
        nbrs_ts_affected = tf.math.count_nonzero(a_t_vector, axis=1)

        # Decide whether enough time stamps are abnormal for the whole example being classified as anomaly, Formula 7
        # Cast to ensure compatibly, hyper parameters may contain a float
        is_anomaly = tf.math.greater(nbrs_ts_affected, tf.cast(nbr_affected_ts_threshold, dtype=tf.int64))

        return is_anomaly, a_t_vector

    @staticmethod
    def divide_with_0_handling(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c

    @staticmethod
    def symptom_score_calculation(base_output: tf.Tensor, a_t_vectors: tf.Tensor):
        """
        Compresses the anomaly scores of each example into a 1D Vector of symptom scores for each feature. Formula 8

        :param base_output: The output of the MTAD_GAT model
        :param a_t_vectors: Boolean tensor which timestamps are affected. Created by anomaly_decision()
        :return: A 1D Vector with shape [features]
        """

        # Convert from tensors to numpy arrays and boolean vector into numerical
        a_t_vectors_float = 1 * a_t_vectors.numpy()
        base_output = base_output.numpy()

        # Extension of the last dimension so that the base_output,
        # so that this can be applied to the base_output element wise, to set scores of non anomaly timestamps to zero
        # -> do not influence the summation over all time stamps
        a_t_vectors_elem_wise = np.repeat(a_t_vectors_float[:, :, np.newaxis], base_output.shape[2], axis=2)

        # Do the same thing to a vector that stores who many time stamps where relevant for a single example
        # in order to weight the sum
        nbr_ts_affected = np.sum(a_t_vectors_float, axis=1)

        # See above
        nbr_ts_affected_elem_wise = np.repeat(nbr_ts_affected[:, np.newaxis], base_output.shape[2], axis=1)

        # Sum up the anomaly scores for each feature over all affected timestamps
        # and weight it by the number of affected timestamps
        only_relevant_timestamp = np.multiply(base_output, a_t_vectors_elem_wise)
        reduced_to_features = np.sum(only_relevant_timestamp, axis=1)

        weighted_by_nbr = Framework.divide_with_0_handling(reduced_to_features, nbr_ts_affected_elem_wise)
        ss_vector = weighted_by_nbr

        return ss_vector

    @staticmethod
    def batch_wrapper(method, batch):
        """
        Wrapper method for symptom identification and tracing because those are defined on an example basis

        :param method: The method that should be executed for each example
        :param batch: The batch of examples
        :return: A list of return values of the method called with each example
        """

        return [method(example) for example in batch]

    def symptom_identification(self, ss_vector):
        """

        :param ss_vector: Output of symptom_score_calculation()
        """

        return SymptomIdentification.symptom_identification(ss_vector,
                                                            self.ontology.feature_names,
                                                            self.hyper.si_mode,
                                                            self.ss_percentiles_train,
                                                            self.hyper.si_parameter)

    def symptom_tracing(self, ss_vector):
        """
        Formula 9

        :param ss_vector: Output of symptom_score_calculation()
        :return: Component labels sorted the highest chance of being affected based on the symptom scores and the feature
            relevance form the ontology and its calculated score as a tuple
        """
        components = np.array(list(self.ontology.relevance_knowledge_str.keys()))
        scores = np.zeros(components.shape)
        comp_sums = np.array([np.sum(self.ontology.relevance_knowledge_scores.get(comp)) for comp in components])

        for i, comp in enumerate(components):
            relevance_score_comp = self.ontology.relevance_knowledge_scores.get(comp)
            scores[i] = np.sum(np.multiply(ss_vector, relevance_score_comp)) / comp_sums[i]
        ranking = scores.argsort()[::-1]
        scores = scores[ranking]
        components = components[ranking]

        return components, scores

    def generate_percentiles_file(self, called_from_train=False):
        # Check whether the percentile files needs to be generated based on the SI modes that will be tested
        if not self.hyper.si_mode in ['example_percentile', 'train_percentile', 'jenks_percentile']:
            return

        # Check if all necessary parameters are set in the hyperparameters, else skip
        if any([self.hyper.gamma is None,
                self.hyper.single_timestamp_anomaly_threshold is None,
                self.hyper.affected_timestamps_threshold is None]):

            if called_from_train:
                print('Not all necessary parameters set for generating the percentiles file. Skipping ...\n')
                return
            else:
                raise ValueError('Not all necessary parameters set for generating the percentiles file.')

        print('Generating percentiles file for this model ...')

        arrays = [self.dataset.x_train, self.dataset.full_next_train, np.arange(self.dataset.num_train_instances)]

        x, full_next, indices = self.dataset.create_batches(self.hyper.batch_size, arrays)

        ss_vectors_batches = []

        for step, (x_batch, full_next, indices_batch) in enumerate(zip(x, full_next, indices)):
            base_output = self.model([x_batch, full_next], training=False)

            are_anomaly, a_t_vectors = self.anomaly_decision(base_output)
            ss_vectors = self.symptom_score_calculation(base_output, a_t_vectors)
            ss_vectors_batches.append(ss_vectors)

        ss_vectors = np.concatenate(ss_vectors_batches, axis=0)

        percentiles_per_feature = []

        for i in range(0, 100):
            percentiles_i = np.percentile(ss_vectors, i, axis=0).reshape(1, -1)
            percentiles_per_feature.append(percentiles_i)

        percentiles_per_feature = np.concatenate(percentiles_per_feature, axis=0)
        self.reset_state()

        np.save(self.config.directory_model_to_use + self.config.ss_percentiles_train_file, percentiles_per_feature)
        print('Saved percentiles file to {}'.format(
            self.config.directory_model_to_use + self.config.ss_percentiles_train_file))
        print()

        self.ss_percentiles_train = percentiles_per_feature
