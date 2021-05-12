import os
import sys

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from execution.GridSearch import grid_search
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from configuration.Enums import GridSearchMode
from framework.Dataset import Dataset
from framework.Framework import Framework


def main():
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'anomalyDetection', training=True)
    checker.pre_init_checks()

    framework = Framework(config, dataset, True)

    if config.use_grid_search:
        framework.config.percentile_calc_after_training = False

    framework.train_model()

    framework = Framework(config, dataset, False)

    if config.use_grid_search:
        if config.grid_search_mode in [GridSearchMode.AD_MODEL_SELECTION, GridSearchMode.AD_AND_FPD_SELECTION]:
            grid_search(framework, config.grid_search_mode)
        else:
            raise ValueError('Cant execute grid search in {} mode when training.'.format(config.grid_search_mode))
    else:
        framework.test_model(print_results=True)


# Allows the combined execution of a training with subsequent evaluation or grid search.
if __name__ == '__main__':
    main()
