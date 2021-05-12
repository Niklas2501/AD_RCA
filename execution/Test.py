import os
import sys

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from execution.GridSearch import grid_search
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from framework.Dataset import Dataset
from framework.Framework import Framework


def main():
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'anomalyDetection', training=False)
    checker.pre_init_checks()

    framework = Framework(config, dataset, training=False)

    if config.use_grid_search:
        grid_search(framework, config.grid_search_mode)
    else:
        framework.test_model(print_results=True, generate=True)


if __name__ == '__main__':
    main()
