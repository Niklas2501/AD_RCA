import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def main():
    pd.set_option('display.max_colwidth', None)

    config = Configuration()

    # import datasets
    y_train = np.load(config.training_data_folder + "train_labels.npy")
    y_test = np.load(config.training_data_folder + "test_labels.npy")

    # get unqiue classes and the number of examples in each
    y_train_single, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_single, y_test_counts = np.unique(y_test, return_counts=True)

    # create a dataframe
    x = np.stack((y_train_single, y_train_counts)).transpose()
    x = pd.DataFrame.from_records(x)

    y = np.stack((y_test_single, y_test_counts)).transpose()
    y = pd.DataFrame.from_records(y)

    x = x.merge(y, how='outer', on=0)
    x = x.rename(index=str, columns={0: 'Failure mode', '1_x': 'Train', '1_y': 'Test'})

    # convert column types in order to be able to sum the values
    x['Train'] = pd.to_numeric(x['Train']).fillna(value=0).astype(int)
    x['Test'] = pd.to_numeric(x['Test']).fillna(value=0).astype(int)
    x['Total'] = x[['Test', 'Train']].sum(axis=1)

    has_test_val = os.path.isfile(config.training_data_folder + "test_val_labels.npy")
    has_train_val = os.path.isfile(config.training_data_folder + "train_val_labels.npy")

    if has_test_val:
        y_val = np.load(config.training_data_folder + "test_val_labels.npy")
        y_val_single, y_val_counts = np.unique(y_val, return_counts=True)
        v = np.stack((y_val_single, y_val_counts)).transpose()
        v = pd.DataFrame.from_records(v, columns=['Failure mode', 'Test Val'])

        x = x.merge(v, how='outer', on='Failure mode')
        x['Test Val'] = pd.to_numeric(x['Test Val']).fillna(value=0).astype(int)
    else:
        x['Test Val'] = 0

    if has_train_val:
        y_val = np.load(config.training_data_folder + "train_val_labels.npy")
        y_val_single, y_val_counts = np.unique(y_val, return_counts=True)
        v = np.stack((y_val_single, y_val_counts)).transpose()
        v = pd.DataFrame.from_records(v, columns=['Failure mode', 'Train Val'])

        x = x.merge(v, how='outer', on='Failure mode')
        x['Train Val'] = pd.to_numeric(x['Train Val']).fillna(value=0).astype(int)

    else:
        x['Train Val'] = 0

    x['Total'] = x[['Train', 'Test', 'Train Val', 'Test Val']].sum(axis=1)
    x = x[['Failure mode', 'Train', 'Train Val', 'Test', 'Test Val', 'Total']]
    x = x.set_index('Failure mode')

    # Combine cases to components
    rows = ['no_failure'] + list(config.component_to_class.keys())
    x_components = pd.DataFrame(0, index=rows, columns=x.columns)
    x_components.index.name = 'Components'
    x_components.loc['no_failure'] = x.loc['no_failure']

    # Sum case values for the corresponding components
    for component in config.component_to_class.keys():
        train, test, test_val, train_val = 0, 0, 0, 0
        for label in config.component_to_class.get(component):

            # Only access values if the failure label exists
            if label in x.index:
                train += x.loc[label, 'Train']
                test += x.loc[label, 'Test']
                test_val += x.loc[label, 'Test Val']
                train_val += x.loc[label, 'Train Val']

        # Insert summed values for thi s component
        cols = ['Train', 'Train Val', 'Test', 'Test Val', 'Total']
        new_values = [train, train_val, test, test_val, train + test_val + test + train_val]

        x_components.loc[component, cols] = new_values

        # print the information to console
    print('----------------------------------------------')
    print('Train and test data sets:')
    print('----------------------------------------------')
    print(x)
    print('----------------------------------------------')
    print('Aggregated by components:')
    print('----------------------------------------------')
    print(x_components)
    print('----------------------------------------------')
    print('Total sum in train:', x['Train'].sum(axis=0))
    print('Total sum in train validation:', x['Train Val'].sum(axis=0))
    print('Total sum in test:', x['Test'].sum(axis=0))
    print('Total sum in test validation:', x['Test Val'].sum(axis=0))
    print('Total sum examples:', x['Total'].sum(axis=0))
    print('----------------------------------------------')

    # print(x_components.to_latex(label ='tab:dataset'))

    # if not os.path.isfile(config.case_base_folder + "train_labels.npy"):
    #     sys.exit(0)
    #
    # # repeat the process for the case base
    # y_train = np.load(config.case_base_folder + "train_labels.npy")  # labels of the case base
    # y_train_single, y_train_counts = np.unique(y_train, return_counts=True)
    #
    # x = np.stack((y_train_single, y_train_counts)).transpose()
    # x = pd.DataFrame.from_records(x)
    # x = x.rename(index=str, columns={0: 'Failure mode', 1: 'Number of examples'})
    # x['Number of examples'] = pd.to_numeric(x['Number of examples']).fillna(value=0).astype(int)
    # x = x.set_index('Failure mode')
    #
    # print('----------------------------------------------')
    # print('Case base:')
    # print('----------------------------------------------')
    # print(x)
    # print('\nTotal sum examples:', x['Number of examples'].sum(axis=0))
    # print('----------------------------------------------\n')


# display the example distribution of the train and test dataset as well as the case case
if __name__ == '__main__':
    main()
