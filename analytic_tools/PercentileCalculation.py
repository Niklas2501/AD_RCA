import os
import sys

# suppress debugging messages of TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from framework.Dataset import Dataset
from framework.Framework import Framework


# Can be used to manually invoke the percentiles file calculation
def main():
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'anomalyDetection', training=False)
    checker.pre_init_checks()

    framework = Framework(config, dataset, training=False)
    framework.load_model()

    framework.generate_percentiles_file()


if __name__ == '__main__':
    main()
