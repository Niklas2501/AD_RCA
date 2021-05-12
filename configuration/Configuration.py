import json

import pandas as pd

####
# Note: Split into multiple classes only serves to improve clarity.
# Only the Configuration class should be used to access all variables.
# Important: It must be ensured that variable names are only used once.
# Otherwise they will be overwritten depending on the order of inheritance!
# All methods should be added to the Configuration class to be able to access all variables
####
from configuration.Enums import TrainTestSplitMode, GridSearchMode


class GeneralConfiguration:

    def __init__(self):
        ###
        # This configuration contains overall settings that couldn't be match to a specific program component
        ###

        # Specifies the maximum number of cores to be used
        self.max_parallel_cores = 40

        # Path and file name to the specific model that should be used for testing and live classification
        # Folder where the models are stored is prepended below
        self.filename_model_to_use = 'selected_2_mod_0'


class ModelConfiguration:

    def __init__(self):
        ###
        # Hyperparameters
        ###

        # Main directory where the hyperparameter config files are stored
        self.hyper_file_folder = '../configuration/hyperparameter_combinations/model_selection_fine_tuning_results/'
        self.use_hyper_file = True

        # The full path to the hyperparameter file that should be used for training a model
        self.hyper_file = self.hyper_file_folder + 'selected_2_mod_0.json'


class TrainingConfiguration:

    def __init__(self):
        ###
        # This configuration contains all parameters defining the way the model is trained
        ###

        # How many model checkpoints are kept, only used if early stopping is disabled.
        # If early stopping is enabled early stopping + 10 are kept
        self.model_files_stored = 50

        # If true, the folder with temporary models created during training is deleted after saving a copy
        # of the model selected by early stopping
        self.delete_temp_models_after_training = True

        # If true, the training process wil try to create the percentiles file used for the FPD right after training
        # Will only work if all necessary parameters (not only the ones for the neural network) are defined
        self.percentile_calc_after_training = False


class InferenceConfiguration:

    def __init__(self):
        ##
        # Settings and parameters for all inference processes
        ##
        # Notes:
        #   - Folder of used model is specified in GeneralConfiguration

        self.enable_intermediate_output = False
        self.test_with_validation_dataset = False

        # If enabled, the content of the hyperparameter file is printed after the evaluation output
        self.enable_config_output = True

        self.use_grid_search = False
        self.grid_search_mode = GridSearchMode.SI_ST_PARAMETER_SELECTION
        self.grid_file_folder = '../configuration/grid_search_files/'
        self.grid_search_config_file = self.grid_file_folder + ''

        self.k_for_hits_at_k = 3
        self.f_score_beta = 2


class PreprocessingConfiguration:

    def __init__(self):
        ###
        # This configuration contains information and settings relevant for the data preprocessing and dataset creation
        ###

        ##
        # Import and data visualisation
        ##

        self.print_column_names: bool = False

        ##
        # Preprocessing
        ##

        # Value is used to ensure a constant frequency of the measurement time points
        self.resample_frequency = "4ms"  # need to be the same for DataImport as well as DatasetCreation

        # Define the length (= the number of timestamps) of the time series generated
        self.time_series_length = 500

        # Defines the step size of window, e.g. for = 2: Example 0 starts at 00:00:00 and Example 1 at 00:00:02
        # For no overlapping: value = seconds(time_series_length * resample frequency)
        self.overlapping_window_step_seconds = 1

        # Configure the motor failure parameters used in case extraction
        self.split_t1_high_low = True
        self.type1_start_percentage = 0.5
        self.type1_high_wear_rul = 25
        self.type2_start_rul = 25

        # seed for how the train/test data is split randomly
        self.random_seed = 42

        # share of examples used as test set
        self.test_split_size = 0.08

        # Share of examples which is separated from the test set to be used as validation set.
        self.test_val_split_size = 0.30

        # Share of examples which is separated from the train set to be used as validation set.
        self.train_val_split_size = 0.10

        # way examples are split into train and test, see enum class for details
        self.split_mode = TrainTestSplitMode.ANOMALY_DETECTION

        self.rocket_kernels = 5_000


# noinspection PyUnresolvedReferences
class StaticConfiguration:

    def __init__(self, dataset_to_import):
        ###
        # This configuration contains data that rarely needs to be changed, such as the paths to certain directories
        ###

        ##
        # Static values
        ##
        # All of the following None-Variables are read from the config.json file because they are mostly static
        # and don't have to be changed very often

        self.datasets = None

        self.zeroOne, self.intNumbers, self.realValues, self.categoricalValues = None, None, None, None

        # noinspection PyUnresolvedReferences
        self.load_config_json('../configuration/config.json')
        # noinspection PyUnresolvedReferences
        self.load_grid_json(self.grid_search_config_file)

        ##
        # Folders and file names
        ##
        # Note: Folder of used model specified in GeneralConfiguration

        # Dataset prefix
        self.data_folder_prefix = '../data/'

        # Folder where the trained models are saved to during learning process
        self.models_folder = self.data_folder_prefix + 'trained_models/'

        # noinspection PyUnresolvedReferences
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use
        self.directory_model_to_use = self.directory_model_to_use if self.directory_model_to_use.endswith(
            '/') else self.directory_model_to_use + '/'

        # Folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = self.data_folder_prefix + 'training_data/'

        # Folder where the normalisation models should be stored
        self.scaler_folder = self.data_folder_prefix + 'scaler/'

        # Name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # File from which the case information should be loaded, used in dataset creation
        self.case_file = '../configuration/cases.csv'

        self.ontology_file = self.data_folder_prefix + 'knowledge/semantic_model.rdf'
        self.ss_percentiles_train_file = 'train_feature_percentiles.npy'

        self.ts_fresh_features_train = 'ts_fresh_features_train.npy'
        self.ts_fresh_features_val = 'ts_fresh_features_val.npy'
        self.ts_fresh_features_test = 'ts_fresh_features_test.npy'

        self.rocket_features_train = 'rocket_features_train.npy'
        self.rocket_features_val = 'rocket_features_val.npy'
        self.rocket_features_test = 'rocket_features_test.npy'

        # Select specific dataset with given parameter
        # Preprocessing however will include all defined datasets
        self.pathPrefix = self.datasets[dataset_to_import][0]
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # Query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # Define file names for all topics
        self.txt15 = self.pathPrefix + 'raw_data/txt15.txt'
        self.txt16 = self.pathPrefix + 'raw_data/txt16.txt'
        self.txt17 = self.pathPrefix + 'raw_data/txt17.txt'
        self.txt18 = self.pathPrefix + 'raw_data/txt18.txt'
        self.txt19 = self.pathPrefix + 'raw_data/txt19.txt'

        self.topicPressureSensorsFile = self.pathPrefix + 'raw_data/pressureSensors.txt'

        self.acc_txt15_m1 = self.pathPrefix + 'raw_data/TXT15_m1_acc.txt'
        self.acc_txt15_comp = self.pathPrefix + 'raw_data/TXT15_o8Compressor_acc.txt'
        self.acc_txt16_m3 = self.pathPrefix + 'raw_data/TXT16_m3_acc.txt'
        self.acc_txt18_m1 = self.pathPrefix + 'raw_data/TXT18_m1_acc.txt'

        self.bmx055_HRS_acc = self.pathPrefix + 'raw_data/bmx055-HRS-acc.txt'
        self.bmx055_HRS_gyr = self.pathPrefix + 'raw_data/bmx055-HRS-gyr.txt'
        self.bmx055_HRS_mag = self.pathPrefix + 'raw_data/bmx055-HRS-mag.txt'

        self.bmx055_VSG_acc = self.pathPrefix + 'raw_data/bmx055-VSG-acc.txt'
        self.bmx055_VSG_gyr = self.pathPrefix + 'raw_data/bmx055-VSG-gyr.txt'
        self.bmx055_VSG_mag = self.pathPrefix + 'raw_data/bmx055-VSG-mag.txt'


class Configuration(
    PreprocessingConfiguration,
    InferenceConfiguration,
    TrainingConfiguration,
    ModelConfiguration,
    GeneralConfiguration,
    StaticConfiguration,
):

    def __init__(self, dataset_to_import=0):
        PreprocessingConfiguration.__init__(self)
        InferenceConfiguration.__init__(self)
        TrainingConfiguration.__init__(self)
        ModelConfiguration.__init__(self)
        GeneralConfiguration.__init__(self)
        StaticConfiguration.__init__(self, dataset_to_import)

    def load_grid_json(self, file_path):
        if self.use_grid_search:
            with open(file_path, 'r') as f:
                self.parameter_grid = json.load(f)

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.datasets = data['datasets']
        self.prefixes = data['prefixes']
        self.error_descriptions = data['error_descriptions']
        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.categoricalValues = data['categoricalValues']

        self.features_used = sorted(data['features_used'])

        self.label_renaming_overall = data['label_renaming_overall']
        self.label_renaming = data['label_renaming']
        self.transfer_from_train_to_test = data['transfer_from_train_to_test']
        self.unused_labels = data['unused_labels']

        self.component_to_class: dict = data['component_to_class']
        self.component_symptom_selection: dict = data['component_symptom_selection']

        # Remove duplicates to ensure output is correct (would result in wrong sum of changed examples otherwise)
        self.unused_labels = list(set(self.unused_labels))
        self.transfer_from_train_to_test = list(set(self.transfer_from_train_to_test))

        # Add inverse dictionary
        self.class_to_component = {}

        for key, values in self.component_to_class.items():
            for value in values:
                self.class_to_component[value] = key

    # return the error case description for the passed label
    def get_error_description(self, error_label: str):
        return self.error_descriptions[error_label]

    # import the timestamps of each dataset and class from the cases.csv file
    def import_timestamps(self):
        datasets = []
        number_to_array = {}

        with open(self.case_file, 'r') as file:
            for line in file.readlines():
                parts = line.split(',')
                parts = [part.strip(' ') for part in parts]
                # print("parts: ", parts)
                # dataset, case, start, end = parts
                dataset = parts[0]
                case = parts[1]
                start = parts[2]
                end = parts[3]
                failure_time = parts[4].rstrip()

                timestamp = (gen_timestamp(case, start, end, failure_time))

                if dataset in number_to_array.keys():
                    number_to_array.get(dataset).append(timestamp)
                else:
                    ds = [timestamp]
                    number_to_array[dataset] = ds

        for key in number_to_array.keys():
            datasets.append(number_to_array.get(key))

        self.cases_datasets = datasets

    def change_current_model(self, model_name: str, models_dir_suffix='trained_models/'):
        def check_end(path):
            return path if path.endswith('/') else path + '/'

        model_name = check_end(model_name)
        models_dir_suffix = check_end(models_dir_suffix)
        self.data_folder_prefix = check_end(self.data_folder_prefix)

        self.filename_model_to_use = model_name
        self.models_folder = self.data_folder_prefix + models_dir_suffix
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use


def gen_timestamp(label: str, start: str, end: str, failure_time: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')
    if failure_time != "no_failure":
        failure_as_time = pd.to_datetime(failure_time, format='%Y-%m-%d %H:%M:%S')
    else:
        failure_as_time = ""

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time, failure_as_time
