import itertools
import os
import sys
from enum import Enum

import pandas  as pd
from numpy import zeros, arange
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from execution.Evaluator import Evaluator
from framework.Dataset import Dataset
from Representations import TSFreshRepresentation, RocketRepresentation


def grid_to_lists(parameter_grid: dict):
    parameter_names, values = [], []
    for key, value in parameter_grid.items():
        parameter_names.append(key)
        values.append(value)

    grid_configurations = list(itertools.product(*values))

    return grid_configurations, parameter_names


def set_parameters(clf, parameter_names, config):
    for parameter, value in zip(parameter_names, config):

        # Ensure the key in hyperparameter defined in the json file exists as a hyperparameter
        if hasattr(clf, parameter):
            setattr(clf, parameter, value)
        else:
            raise ValueError('Unknown parameter: {}'.format(parameter))


def main():
    config = Configuration()

    #####################################################
    # Baseline Configuration
    #####################################################
    class Type(Enum):
        OC_SVM = 0
        ISO_FOREST = 1

    class DatasetType(Enum):
        STANDARD = 0
        TS_FRESH = 1
        ROCKET = 2

    baseline_type = Type.ISO_FOREST
    enable_intermediate_output = False
    dataset_type = DatasetType.TS_FRESH

    oc_svm_parameters = {
        # 1 / (61 * np.var(dataset.x_train)) = 0.07553405212173285
        'gamma': ['scale'],
        "nu": [0.03],

        # static parameters defined in case the default changes
        'kernel': ['rbf'],
        'shrinking': [True],  # Did not result in any differences during first test
        'tol': [1e-3],
        'verbose': [False],
        'max_iter': [-1],
        'cache_size': [8000]
    }

    iso_forest_parameters = {
        'n_estimators': [50],
        'max_features': [0.05],
        'max_samples': [64],
        'contamination': [0.0],

        # static parameters
        'bootstrap': [True],
        'random_state': [23],
        'verbose': [0],
        'n_jobs': [config.max_parallel_cores]
    }

    #####################################################

    config.enable_intermediate_output = False
    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    if dataset_type == DatasetType.ROCKET:
        print('Using the Rocket representation of the data as input\n')
        rep = RocketRepresentation(config, dataset)
        dataset = rep.load_into_dataset()
    elif dataset_type == DatasetType.TS_FRESH:
        print('Using the TSFresh representation of the data as input\n')
        rep = TSFreshRepresentation(config, dataset)
        dataset = rep.load_into_dataset()
    else:
        print('Using the standard data as input\n')

    x_train = dataset.x_train
    if config.test_with_validation_dataset:
        x_test = dataset.x_test_val
    else:
        x_test = dataset.x_test

    set_name = 'validation' if config.test_with_validation_dataset else 'test'

    if dataset_type == DatasetType.STANDARD:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    print(x_train.shape, x_test.shape)

    if baseline_type == Type.OC_SVM:
        parameter_grid = oc_svm_parameters
        baseline_type_str = 'OneClassSVM'
    elif baseline_type == Type.ISO_FOREST:
        parameter_grid = iso_forest_parameters
        baseline_type_str = 'IsolationForest'
    else:
        raise ValueError('Unknown baseline type defined: {}'.format(baseline_type))

    grid_configurations, parameter_names = grid_to_lists(parameter_grid)
    at_vector_placeholder = zeros(shape=(500,)) # must be passed to evaluator
    results = []

    print('Evaluating {} on the {} dataset with {} parameter configurations.\n'.format(baseline_type_str, set_name,
                                                                                       len(grid_configurations)))

    for index, parameters in enumerate(grid_configurations):
        if baseline_type == Type.OC_SVM:
            clf = OneClassSVM()
        elif baseline_type == Type.ISO_FOREST:
            clf = IsolationForest()
        else:
            raise ValueError('Unknown baseline type defined: {}'.format(baseline_type))

        evaluator = Evaluator(config, dataset)
        set_parameters(clf, parameter_names, parameters)

        # print('Evaluating One Class SVM on the {} dataset. \n'.format(set_name))
        # print('Parameters used:', clf.get_params())
        print('Evaluating {} with configuration {}/{}'.format(baseline_type_str, index + 1, len(grid_configurations)))
        clf.fit(x_train)
        y_test_pred = clf.predict(x_test)

        for index, prediction in enumerate(y_test_pred):
            is_anomaly = False if prediction < 0 else True
            evaluator.add_ad_result(is_anomaly, at_vector_placeholder, index, is_anomaly)

        evaluator.calculate_results()

        if enable_intermediate_output:
            evaluator.print_baseline_results(baseline_type_str, clf.get_params())
            print('\n')

        result = evaluator.ad_results_final.drop(columns=['AVG # affected', '#Examples'])

        params: dict = clf.get_params()

        # Remove constant parameters that do not need to be displayed in the log
        for c in ['behaviour', 'n_jobs', 'random_state', 'verbose']:
            if c in params.keys():
                params.pop(c)

        result.loc['combined', 'Parameters'] = str(params)
        results.append(result)

    cols = results[0].columns
    combined_results = pd.DataFrame(columns=cols)

    for parameters, result in zip(grid_configurations, results):
        combined_row = result.loc['combined', cols]
        combined_results.loc[len(combined_results), cols] = combined_row

    combined_results = combined_results.sort_values(by=["F1", "F2", "TPR", "Prec"], ascending=False)

    print()
    print('Grid search results:')
    print(combined_results.drop(columns=['TP', 'FP', 'TN', 'FN']).to_string())
    print('\n')
    print('Best result in detail:')
    print(combined_results.iloc[0].name)


if __name__ == '__main__':
    main()
