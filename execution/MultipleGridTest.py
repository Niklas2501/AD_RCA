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
from configuration.Enums import GridSearchMode


def main():
    model_dirs = ['test_1', 'test_2']
    sub_dir = 'trained_models/sub_dir'

    for model_dir in model_dirs:
        print('--------------------------------------------------------------------------------------------------')
        print(model_dir)
        print('--------------------------------------------------------------------------------------------------')

        config = Configuration()

        if sub_dir is None or sub_dir == '':
            config.change_current_model(model_dir)
        else:
            config.change_current_model(model_dir, sub_dir)

        dataset = Dataset(config.training_data_folder, config)
        dataset.load(print_info=False)

        checker = ConfigChecker(config, dataset, 'anomalyDetection', training=False)
        checker.pre_init_checks()

        framework = Framework(config, dataset, training=False)

        grid_search(framework, GridSearchMode.AD_MODEL_SELECTION)
        print('\n')


# Can be used to evaluate multiple models with the same test / grid search configuration.
if __name__ == '__main__':
    main()
