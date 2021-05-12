import itertools
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from configuration.Enums import GridSearchMode
from framework.Framework import Framework


def set_current_hps(framework: Framework, hyperparameters: [str], config):
    # Set the hyperparameters in the hyperparameters object used during testing

    for hyperparameter, value in zip(hyperparameters, config):

        # Ensure the key in hyperparameter defined in the json file exists as a hyperparameter
        if hasattr(framework.hyper, hyperparameter):
            setattr(framework.hyper, hyperparameter, value)
        else:
            raise ValueError('Unknown hyperparameter: {}'.format(hyperparameter))


def grid_search(framework: Framework, mode: GridSearchMode):
    framework.config.enable_intermediate_output = False
    framework.config.enable_config_output = False

    parameter_grid: dict = framework.config.parameter_grid
    si_st_params = ['relevance_mapping', 'unaffected_component_threshold', 'si_mode', 'si_parameter']
    grid_search_params = ['notes', 'ad_metrics', 'si_st_metrics', 'sorted_by']
    percentile_based_modes = ['train_percentile', 'jenks_percentile']

    if mode == GridSearchMode.AD_MODEL_SELECTION:
        print('Grid search running in anomaly detection model selection mode.')
        print('The following hyperparameters will not be evaluated:')
        print(*si_st_params, sep='\n')
        print()

        # Remove hyperparameter that can't / should not be evaluated when using this mode
        for param in si_st_params:
            parameter_grid.pop(param, None)

        # Force some hp values that are necessary but should not be altered
        # Setting the first one to None will disable the false positive detection so the displayed result will only
        # depend on the trained model
        parameter_grid['unaffected_component_threshold'] = [None]
        parameter_grid['si_mode'] = ['sort_only']
        parameter_grid['si_parameter'] = [0]
        parameter_grid['relevance_mapping'] = [{
            "h": 1.0,
            "m": 0.3,
            "l": 0.0,
            "e": 0.0
        }]

    elif mode == GridSearchMode.AD_AND_FPD_SELECTION:
        print('Grid search running in AD and FPD selection mode.')
        print('The following hyperparameters will not be evaluated:')
        print('si_mode')
        print('si_parameter\n')

        # Check if the necessary parameters are set
        for key in ['unaffected_component_threshold', 'relevance_mapping']:
            value = parameter_grid[key]
            if len(value) == 0 or value[0] is None:
                print('Parameter {} not defined in the grid search configuration file.'.format(key))
                print('To be on the safe side, the process is aborted.')
                sys.exit()

        # Force some hp values that are necessary but should not be altered
        parameter_grid['si_mode'] = ['sort_only']
        parameter_grid['si_parameter'] = [0]

    elif mode == GridSearchMode.SI_ST_PARAMETER_SELECTION:
        print('Grid search running in SI / ST parameter selection mode.')
        print('Only the following hyperparameters will be evaluated:')
        print(*si_st_params, sep='\n')
        print()
        print('WARNING: Does not check whether all other hyperparameters are set for the current model.')
        print('Crashes will likely be the result of this not been done after AD model selection grid search.')
        print()

        # Extract the relevant hyperparameters, stop if relevant one is not contained
        try:
            parameter_grid = {param: parameter_grid[param] for param in si_st_params + grid_search_params}
        except KeyError:
            print('Relevant parameter not defined in the grid search configuration file.')
            print('To be on the safe side, the process is aborted.')
            sys.exit()

        # Additional check if list-type hyperparameters are empty = not set correctly
        for key in si_st_params:
            value = parameter_grid[key]
            if len(value) == 0:
                print('Parameter {} not defined in the grid search configuration file.'.format(key))
                print('To be on the safe side, the process is aborted.')
                sys.exit()

        # Check whether the percentile files needs to be generated based on the SI modes that will be tested
        generation_necessary = False
        modes_tested = parameter_grid.get('si_mode')
        for mode in modes_tested:
            if mode in percentile_based_modes:
                generation_necessary = True
                break

        if generation_necessary:
            framework.generate_percentiles_file()

        if (generation_necessary or 'example_percentile' in modes_tested):
            if parameter_grid.get('si_parameter') == 'generate':
                parameter_grid['si_parameter'] = [i for i in range(1, 99, 2)]
        else:
            parameter_grid.pop('si_parameter')
    else:
        raise ValueError('Unknown grid search mode passed:', mode)

    # Metrics displayed in the results
    ad_metrics = parameter_grid.get('ad_metrics')
    si_st_metrics = parameter_grid.get('si_st_metrics')

    # Prepend component names to be able to distinguish metrics used in both
    ad_metric_cols = ['AD ' + metric for metric in ad_metrics]
    si_st_metric_cols = ['SI/ST ' + metric for metric in si_st_metrics]

    hyperparameters = []
    values = []

    # Creation of lists with varied parameters and their values,
    # Leave out non-parameter and empty entries in the json file (only relevant for the ad selection mode)
    for key, value in parameter_grid.items():
        if key not in ["notes", "ad_metrics", "si_st_metrics", "sorted_by"] and len(value) != 0:
            hyperparameters.append(key)
            values.append(value)

    # Create all possible combinations
    grid_configurations = list(itertools.product(*values))
    nbr_configurations = len(grid_configurations)

    # Create dataframe to store the results
    cols = hyperparameters + ad_metric_cols + si_st_metric_cols
    results = pd.DataFrame(columns=cols)
    results.index.name = 'Comb'

    # Output which model is used.
    set_name = 'validation' if framework.config.test_with_validation_dataset else 'test'
    print('Testing {} combinations via grid search '.format(nbr_configurations))
    print('for model {} on the {} dataset. \n'.format(framework.config.filename_model_to_use, set_name))

    start = perf_counter()

    result_dfs_ad = []
    results_dfs_ad_inter = []
    result_dfs_si_st = []

    for index, config in enumerate(grid_configurations):
        # print('Currently testing combination {}/{} {} ...'.format(index, nbr_configurations - 1, config))
        set_current_hps(framework, hyperparameters, config)

        # Execute the evaluation for the current configuration and store  relevant metrics and the resulting dataframes
        evaluator = framework.test_model(print_results=False)
        results_ad = evaluator.ad_results_final.loc['combined', ad_metrics]
        results_si_st = evaluator.si_st_results.loc['combined', si_st_metrics]
        result_row = list(config) + list(results_ad) + list(results_si_st)
        results.loc[len(results)] = result_row

        result_dfs_ad.append(evaluator.ad_results_final.copy(deep=True))
        results_dfs_ad_inter.append(evaluator.ad_results_intermed.copy(deep=True))
        result_dfs_si_st.append(evaluator.si_st_results.copy(deep=True))

    end = perf_counter()

    # Drop forced = static hyperparameters depending on execution mode
    if mode == GridSearchMode.AD_MODEL_SELECTION:
        results = results.drop(
            columns=['unaffected_component_threshold', 'relevance_mapping', 'si_mode', 'si_parameter'])
    elif mode == GridSearchMode.AD_AND_FPD_SELECTION:
        results = results.drop(columns=['si_mode', 'si_parameter'])

    # results = results.fillna(-999)

    # higher = better is assumed
    sorting_cols = list(np.array((ad_metric_cols + si_st_metric_cols))[parameter_grid.get('sorted_by')])
    results = results.sort_values(by=sorting_cols, ascending=False)

    # Truncate column names if too large
    limit = 20
    results_out = results.copy()
    results_out.columns = [(col[:limit - 2] + '..') if len(col) > limit - 2 else col for col in results_out.columns]
    best_comb_index = results.index[0]

    print()
    print('Best combinations tested:')
    print(results_out.head(150).to_string())
    print()
    print('Full result output for the best combination:')
    print(results_dfs_ad_inter[best_comb_index].to_string())
    print()
    print(result_dfs_ad[best_comb_index].to_string())
    print()
    print(result_dfs_si_st[best_comb_index].to_string())
    print()
    print('Execution time: {}'.format(end - start))
