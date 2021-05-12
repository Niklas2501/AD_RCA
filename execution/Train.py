import os
import sys

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from framework.Dataset import Dataset
from framework.Framework import Framework


def main():
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'anomalyDetection', training=True)
    checker.pre_init_checks()

    framework = Framework(config, dataset, True)
    framework.train_model()


if __name__ == '__main__':
    main()
