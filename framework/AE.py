import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions

from configuration.Hyperparameter import Hyperparameters


##
# Implementation based on: https://bit.ly/3oRMiQz
##

class Sampling(layers.Layer):

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def __init__(self, latent_dim, intermediate_dim, name="vae_encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)

    def call(self, inputs, **kwargs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        return z_mean, z_log_var


class Decoder(layers.Layer):

    def __init__(self, original_dim, intermediate_dim, name="vae_decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(original_dim)
        self.dense_log_var = layers.Dense(original_dim)

    def call(self, inputs, **kwargs):
        """
        This variant of a VAE decoder returns the mean and variance with which X hat can be sampled.
        Necessary for the reconstruction probability calculation.
        """

        inter = self.dense_proj(inputs)
        x_mean = self.dense_mean(inter)
        x_log_var = self.dense_log_var(inter)
        return x_mean, x_log_var


class VAE(layers.Layer):

    def __init__(self, hyper: Hyperparameters, name="VAE", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)

        self.hyper = hyper

        self.latent_dim = self.hyper.d3_vae_latent_dim
        self.intermediate_dim = self.hyper.d3_vae_inter_dim

        # If the variant is used, where the gru output is reconstructed, the output size of the gru layer is fixed
        # to the time series depth, so we use this value instead of the value configured in d1_gru_units
        # which could be different
        if 'reconstruct_gru' in self.hyper.variants:
            target_feature_dim = self.hyper.time_series_depth
        else:
            target_feature_dim = self.hyper.d1_gru_units[-1]

        self.gru_output_dim = (self.hyper.time_series_length, target_feature_dim)
        self.gru_output_dim_flat = self.hyper.time_series_length * target_feature_dim

        self.rec_target_dim = (self.hyper.time_series_length, self.hyper.time_series_depth)
        self.rec_target_dim_flat = self.hyper.time_series_length * self.hyper.time_series_depth

        self.flatten_gru_output = tf.keras.layers.Reshape(target_shape=(self.gru_output_dim_flat,))
        self.flatten_rec_target = tf.keras.layers.Reshape(target_shape=(self.rec_target_dim_flat,))
        self.rebuild_rec_target_shape = tf.keras.layers.Reshape(target_shape=self.rec_target_dim)

        self.encoder = Encoder(latent_dim=self.latent_dim, intermediate_dim=self.intermediate_dim)
        self.decoder = Decoder(original_dim=self.rec_target_dim_flat, intermediate_dim=self.intermediate_dim)

        self.sampling = Sampling()

        print('Adding VAE as reconstruction model with dimensions {} & {} and ...'.format(self.intermediate_dim,
                                                                                          self.latent_dim))

    def call(self, inputs, **kwargs):
        inputs = self.flatten_gru_output(inputs)
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        x_mean, x_log_var = self.decoder(z)
        x_reconstructed = self.sampling([x_mean, x_log_var])
        x_reconstructed = self.rebuild_rec_target_shape(x_reconstructed)

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)

        return x_reconstructed

    @tf.function
    def reconstruction_probability(self, gru_output, reconstruction_target, L=None):
        reconstruction_target = self.flatten_rec_target(reconstruction_target)
        gru_output = self.flatten_gru_output(gru_output)
        z_means, z_log_vars = self.encoder(gru_output)

        # Sample size is set to the batch size if no specific value is passed
        L = gru_output.shape[0] if L is None else L
        L = tf.constant(L)
        sample_index = tf.constant(0)
        sampling_results = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Track the sampling process
        def while_condition(sample_index, L, _):
            return tf.less(sample_index, L)

        # Note to self: all parameters of body must be returned in the same order
        # Write the result of a single sample calculation to the result array
        def while_body(sample_index, L, results):
            single_result = single_sample_calculation()
            results = results.write(sample_index, single_result)
            return sample_index + 1, L, results

        def single_sample_calculation():
            """
            One call calculates one of L Samples for each example in the batch
            Based on An and Chao http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf,
            but applied to a single cell (time step, feature)
            :return:
            """

            # Sample x' distribution for this sample
            z = self.sampling([z_means, z_log_vars])
            x_mean, x_log_var = self.decoder(z)

            # Convert log. variance to standard deviation
            x_sigma = tf.sqrt(tf.exp(x_log_var) + 1e-5)

            # Split the calculation for this sample for each example in the batch
            reconstruction_probs = tf.vectorized_map(self.each_example, [reconstruction_target, x_mean, x_sigma])
            return reconstruction_probs

        # Execute a loop the calculate the results for L samples
        # Note to self: return values correspond to the loop vars
        _, _, sampling_results = tf.while_loop(cond=while_condition, body=while_body,
                                               loop_vars=[sample_index, L, sampling_results])

        # Combine the sample results into a tensor and calculate the mean of each cell over the L samples
        # sampling_results = sampling_results.stack()
        sampling_results = sampling_results.gather(range(0, L))
        sampling_results = tf.stack(sampling_results, axis=0)
        sampling_results = tf.reduce_mean(sampling_results, axis=0)

        # Reshape from 1 dimensional vector the time series dimensions (time series length x time series depth)
        sampling_results = self.rebuild_rec_target_shape(sampling_results)

        return sampling_results

    def each_example(self, args):
        x, means, sigmas = args

        # Split the calculation from each example in the batch into a cell based calculation
        reconstruction_prob = tf.vectorized_map(self.each_cell, [x, means, sigmas], fallback_to_while_loop=True)
        return reconstruction_prob

    @tf.function
    def each_cell(self, args):
        reconstructed_cell, mean_sample, sigma_sample = args

        # Calculate the probability of the cell value being reconstructed from a normal distribution defined by
        # the passed mean and standard deviation
        normal = distributions.Normal(mean_sample, sigma_sample, validate_args=False, allow_nan_stats=False)

        # Changed from probability density to cumulative density function in order to return a probability in [0,1]
        reconstruction_prob_for_cell = normal.cdf(reconstructed_cell)

        return reconstruction_prob_for_cell


class GRU_AE(layers.Layer):

    def __init__(self, hyper: Hyperparameters, name="GRU_AE", **kwargs):
        super(GRU_AE, self).__init__(name=name, **kwargs)

        self.hyper = hyper

        self.inter_dim = self.hyper.d3_vae_inter_dim
        self.latent_dim = self.hyper.d3_vae_latent_dim
        self.output_dim = self.hyper.time_series_depth

        # If the variant is used, where the gru output is reconstructed, the output size of the gru layer is fixed
        # to the time series depth, so we use this value instead of the value configured in d1_gru_units
        # which could be different
        if 'reconstruct_gru' in self.hyper.variants:
            self.input_dim = self.hyper.time_series_depth
        else:
            self.input_dim = self.hyper.d1_gru_units[-1]

        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(layers.Input(shape=(self.hyper.time_series_length, self.input_dim)))

        if self.inter_dim is not None:
            self.encoder.add(layers.GRU(units=self.inter_dim, return_sequences=True))

        self.encoder.add(layers.GRU(units=self.latent_dim, return_sequences=True))

        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(layers.Input(shape=(self.hyper.time_series_length, self.latent_dim)))

        if self.inter_dim is not None:
            self.decoder.add(layers.GRU(self.inter_dim, return_sequences=True))

        self.decoder.add(layers.GRU(self.output_dim, return_sequences=True))

        print('Adding GRU AE as reconstruction model with dimensions {} & {} and ...'.format(self.inter_dim,
                                                                                             self.latent_dim))

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        inputs_reconstructed = self.decoder(z)
        return inputs_reconstructed
