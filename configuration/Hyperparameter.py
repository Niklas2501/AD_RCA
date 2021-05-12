import json


class Hyperparameters:

    def __init__(self):

        ##
        # Important: Variable names must match json file entries
        ##

        # Needs to be changed after dataset was loaded
        self.time_series_length = None
        self.time_series_depth = None

        # General information that can be passed on to the saved file via a string
        self.notes = None

        # Defines the model structure
        self.variants = None
        self.defined_variants = ['zhao', 'reconstruct_gru', 'gru_fbm', 'gru_ae', 'reconstruction_error']

        # General hyperparameters
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

        self.early_stopping_enabled = None
        self.early_stopping_limit = None

        self.gradient_cap_enabled = None
        self.gradient_cap = None

        # MTAD-GAT hyperparameters
        self.gamma = None  # h_gamma

        self.conv_filters = None  # h_filter
        self.conv_kernel_size = None  # h_kernel
        self.conv_strides = None

        self.d1_gru_units = None  # h_GRU
        self.d2_fc_units = None  # h_FBM
        self.d3_vae_latent_dim = None  # h_lat

        # MTAD-GAT hyperparameters not mentioned
        self.d3_vae_inter_dim = None  # h_inter
        self.sw_size_time_gat = None  # h_sw

        # Additional hyperparameters for SI / ST
        self.single_timestamp_anomaly_threshold = None  # h_0
        self.affected_timestamps_threshold = None  # h_1
        self.unaffected_component_threshold = None  # h_3
        self.relevance_mapping = None  # h_phi

        self.si_parameter = None  # h_2
        self.si_mode = None

    def set_time_series_properties(self, dataset):
        self.time_series_length = dataset.time_series_length
        self.time_series_depth = dataset.time_series_depth

    # Allows the import of a hyper parameter configuration from a json file
    def load_from_file(self, file_path, use_hyper_file=True):

        if not use_hyper_file:
            return

        file_path = file_path + '.json' if not file_path.endswith('.json') else file_path

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.notes = data['notes']

        # Defines the model structure
        self.variants = data['variants']
        assert all(elem in self.defined_variants
                   for elem in self.variants), 'Unknown variant defined in hyperparameters.'

        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        self.learning_rate = data['learning_rate']

        self.early_stopping_enabled = data['early_stopping_enabled']
        if self.early_stopping_enabled:
            self.early_stopping_limit = data['early_stopping_limit']

        self.gradient_cap_enabled = data['gradient_cap_enabled']
        if self.gradient_cap_enabled:
            self.gradient_cap = data['gradient_cap']

            # Disable gradient cap again if unsuitable value in file
            self.gradient_cap_enabled = True if self.gradient_cap is not None and self.gradient_cap > 0 else False

        self.gamma = data['gamma']
        self.conv_filters = data['conv_filters']
        self.conv_kernel_size = data['conv_kernel_size']
        self.conv_strides = data['conv_strides']

        self.d1_gru_units = data['d1_gru_units']
        self.d2_fc_units = data['d2_fc_units']

        self.d3_vae_inter_dim = data['d3_vae_inter_dim']
        self.d3_vae_latent_dim = data['d3_vae_latent_dim']

        self.sw_size_time_gat = data['sw_size_time_gat']
        self.single_timestamp_anomaly_threshold = data['single_timestamp_anomaly_threshold']
        self.affected_timestamps_threshold = data['affected_timestamps_threshold']
        self.unaffected_component_threshold = data['unaffected_component_threshold']
        self.relevance_mapping = data['relevance_mapping']

        self.si_parameter = data['si_parameter']
        self.si_mode = data['si_mode']

        si_modes = ['sort_only', 'diff_elbow', 'rsm_gan_elbow', 'jenks_breaks', 'example_percentile',
                    'train_percentile', 'jenks_percentile', None]
        assert self.si_mode in si_modes, 'Unknown symptom identification mode defined in hyperparameters:' + self.si_mode

    def write_to_file(self, path_to_file):

        # Creates a dictionary of all class variables and their values
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        with open(path_to_file, 'w') as outfile:
            json.dump(dict_of_vars, outfile, indent=4)

    def print_hyperparameters(self):
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        print(json.dumps(dict_of_vars, sort_keys=False, indent=4))
