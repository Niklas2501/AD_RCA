import tensorflow as tf

from configuration.Hyperparameter import Hyperparameters
from framework.AE import VAE, GRU_AE
from framework.Models import Convolutions, FeatureGAT, TimeGAT, ForecastingModel, GRU, GRU_ForecastingModel


class MTAD_GAT(tf.keras.Model):

    def __init__(self, hyper: Hyperparameters, feature_mask, time_mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper: Hyperparameters = hyper
        self.feature_mask = feature_mask
        self.time_mask = time_mask

        # Instantiate components
        self.convolutions = Convolutions(self.hyper)
        self.feature_gat = FeatureGAT(self.hyper).create_model()
        self.time_gat = TimeGAT(self.hyper).create_model()
        self.gru = GRU(self.hyper)

        if 'gru_fbm' in self.hyper.variants:
            self.forecasting_model = GRU_ForecastingModel(self.hyper)
        else:
            self.forecasting_model = ForecastingModel(self.hyper)

        if 'gru_ae' in self.hyper.variants:
            self.reconstruction_model = GRU_AE(self.hyper)
            self.hyper.variants.append('reconstruction_error')
        else:
            self.reconstruction_model = VAE(self.hyper)

        if 'reconstruct_gru' in self.hyper.variants:
            print('The RBM reconstructs the output of the gru layer.')
        else:
            print('The RBM reconstructs the initial model input.')
        print()

    def call(self, inputs, training=False, **kwargs):
        """
        :param inputs: Batch of examples.
            If not training must contain a pair of [batch, batch full next]
        :param training: Whether the model is currently trained
        :param kwargs:
        :return: If not training a tensor with the same shape as the input containing the
            anomaly scores for each feature at each time stamp. If training the output of the GRU layer
        """

        if not training:
            inputs, next_values = inputs
        else:
            next_values = None

        conv_output = self.convolutions(inputs, training)

        # Input of feature graph conv layer: ([batch], Nodes, Features)
        # Here: Nodes = Attributes (univariate time series), Features = Time steps
        # Current shape: ([batch], Time steps, Attributes), so we must exchange the second and third dimension
        feature_gat_input = tf.transpose(conv_output, perm=[0, 2, 1])
        time_gat_input = conv_output

        feature_gat_output = self.feature_gat([feature_gat_input, self.feature_mask])
        time_gat_output = self.time_gat([time_gat_input, self.time_mask])

        # Reshape back to shape before gat layer
        feature_gat_output = tf.transpose(feature_gat_output, perm=[0, 2, 1])

        gru_input = tf.concat([conv_output, feature_gat_output, time_gat_output], axis=2)
        gru_output = self.gru(gru_input, training=training)

        # The training procedure manually invokes the forecasting and reconstruction model
        if training:
            return gru_output

        reconstruction_target = gru_output if 'reconstruct_gru' in self.hyper.variants else inputs

        if 'reconstruction_error' in self.hyper.variants:
            rec_output = self.reconstruction_model(gru_output)
            rec_output = tf.math.abs(tf.math.subtract(reconstruction_target, rec_output))
            rec_output = tf.math.square(rec_output)
        else:
            rec_output = self.reconstruction_model.reconstruction_probability(gru_output, reconstruction_target)

        for_output = self.forecasting_model(gru_output, training)
        base_output = self.score_calculation(next_values, for_output, rec_output)

        return base_output

    def score_calculation(self, full_next: tf.Tensor, output_for: tf.Tensor, output_rec: tf.Tensor):
        """
        :param full_next:  Tensor with shape [batch size, timestamps, features] where the timestamp with index t
            contains the values at time t+1 compared to the actual input into the model
        :param output_for: The output of the forecasting model with shape [batch size, timestamps, features]
        :param output_rec: The output of the reconstruction model with shape [batch size, timestamps, features]
        :return: A tensor with the same shape as all of the inputs where each value is the anomaly
            score for the specific attribute at a give timestamp
        """

        if 'reconstruction_error' in self.hyper.variants:
            diff_reconstruction = output_rec
        else:
            diff_reconstruction = tf.math.subtract(tf.constant(1.0), output_rec)

        diff_forecasting = tf.math.square(tf.math.subtract(output_for, full_next))
        weighted_prop = tf.math.scalar_mul(scalar=self.hyper.gamma, x=diff_reconstruction)
        numerator = tf.math.add(diff_forecasting, weighted_prop)
        denominator = tf.constant(1.0 + self.hyper.gamma)
        scores = tf.math.divide(numerator, denominator)

        return scores
