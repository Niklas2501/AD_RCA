from enum import Enum


class TrainTestSplitMode(Enum):
    # Examples of a single run to failure are not in both train and test
    ENSURE_NO_MIX = 0

    # Train only consists of no_failure examples, also includes ENSURE_NO_MIX guarantee
    ANOMALY_DETECTION = 1


class GridSearchMode(Enum):
    # Only the basic metrics for the anomaly detection are tested
    AD_MODEL_SELECTION = 0

    # Only 'relevance_mapping', 'unaffected_component_threshold', 'si_mode', 'si_parameter' are evaluated.
    # Requires a trained model, thus can not be used with CombinedTrainTest
    SI_ST_PARAMETER_SELECTION = 1

    # Enables grid search for neural network parameters and ST / FPD parameters 'relevance_mapping' and
    # 'unaffected_component_threshold'
    AD_AND_FPD_SELECTION = 2
