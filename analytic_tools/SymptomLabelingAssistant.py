import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from datetime import datetime
from itertools import cycle, islice
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from configuration.Configuration import Configuration
from framework.Dataset import Dataset


def plot_simple(df: pd.DataFrame, file_name, colors=None, title=None, fig_height=None):
    plt.style.use("bmh")

    # Ensure subplot are distributed over the full plot, don't change the values
    plt.subplots_adjust(top=0.85, bottom=0.05)

    fig_width = 30
    fig_height = fig_height if fig_height is not None else 80
    format = 'png' if fig_height <= 95 else 'svg'

    if format == 'svg' and not file_name.endswith('svg'):
        file_name = file_name + '.svg'
    elif format == 'png' and not file_name.endswith('png'):
        file_name = file_name + '.png'

    if colors is None:
        axes = df.plot(subplots=True, shiarex=True, figsize=(fig_width, fig_height))
    else:
        axes = df.plot(subplots=True, sharex=True, figsize=(fig_width, fig_height), color=colors)

    for ax in axes:
        # Uniform output of the y-axis and the legend position
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 540)
        ax.set_yticks([0.0, 0.5, 1])
        ax.set_yticklabels(["", "0.5", ""])
        ax.legend(loc='center right')

    # Use suptitle as main title so its at the top, don't change y value
    if title is not None:
        plt.suptitle('All sensors: ' + title, fontsize=35, y=0.90)
    else:
        plt.suptitle('All sensors', fontsize=35, y=0.90)

    plt.savefig(file_name, dpi=450, bbox_inches="tight", format=format)


def single_plot(current_label, compared_label,
                current_label_short, compared_label_short,
                current_example_index, compared_example_index,
                mark_relevant, reduce_to_relevant, plot_ts, plot_both, plot_matrix,
                dataset, features):
    current_example = dataset[current_example_index]
    compared_example = dataset[compared_example_index]

    # to suppress possibly undefined error
    symptoms_current_example = None

    # If one of the features is selected the data is read from a file and prepared
    if mark_relevant or reduce_to_relevant:
        import json
        with open('symptom-selection-old.json') as json_file:
            symptom_selection = json.load(json_file)
            symptom_selection: dict = symptom_selection['relevant_features']

        symptoms_current_example = symptom_selection.get(current_label)
        print('Relevant symptoms for ', current_label, ':', symptoms_current_example, '\n')

    if reduce_to_relevant and symptoms_current_example is not None:

        # Reduce both examples to the data streams defined as symptoms for the label
        relevant_indices = np.isin(features, symptoms_current_example)
        current_example = current_example[:, relevant_indices]
        compared_example = compared_example[:, relevant_indices]
        features = features[relevant_indices]

    elif mark_relevant and symptoms_current_example is not None:

        # Prepend r_ as marker to the data stream name displayed in the legend
        renamed = [('r_' + feature if feature in symptoms_current_example else feature) for feature in features]
        features = np.array(renamed)

    current_example_df = pd.DataFrame(data=current_example, columns=features)
    compared_example_df = pd.DataFrame(data=compared_example, columns=features)

    description = ''.join([current_label, '-', str(current_example_index), '-',
                           compared_label, '-', str(compared_example_index), ])
    file_name = ''.join(
        ['../data/visualization/root_cause_labeling_examples/', description])

    if plot_ts:
        print('Creating simple time series plot for', description)
        colors = list(islice(cycle(['b', 'g', 'c', 'r']), None, len(current_example_df)))
        plot_simple(current_example_df, file_name=file_name + '-simple-plot', colors=colors)

    if plot_both:
        print('Creating plot with comparison for', description)

        # Append short label code to feature names and update in dataframes
        current_features = np.array([feature + '_' + current_label_short for feature in features])
        compared_features = np.array([feature + '_' + compared_label_short for feature in features])
        current_example_df.columns = current_features
        compared_example_df.columns = compared_features

        # Alternate the data stream for both examples,
        # e.g: label1_stream1, label2_stream1, label1_stream2, label1_stream2,
        sorted_columns = []
        for cur, com in zip(current_features, compared_features):
            sorted_columns.append(cur)
            sorted_columns.append(com)

        # Combine into single dataframe and sort columns
        combined = pd.concat([current_example_df, compared_example_df], axis=1)
        combined = combined[sorted_columns]

        # Two successive data streams get the same color because same feature
        colors = list(islice(cycle(['b', 'b', 'g', 'g', 'c', 'c', 'r', 'r']), None, len(combined)))

        # Pass info about short label codes to be displayed in plot
        suptitle = current_label + ' = ' + current_label_short + ', ' + compared_label + ' = ' + compared_label_short
        plot_simple(combined, file_name + '-combined', colors, suptitle, fig_height=130)

    if plot_matrix:
        print('Creating scatter matrix for', description)
        scatter_matrix(current_example_df, alpha=0.4, figsize=(60, 60), diagonal='kde')

        plt.savefig(file_name + '-matrix', dpi=500)

        plt.show()

    print()


def plot_single_label(current_label, compared_label,
                      current_label_index, compared_label_index,
                      compared_label_short, from_train, mark_relevant,
                      reduce_to_predefined_relevant, plot_ts, plot_combined, plot_matrix, direct_index):
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    label_to_indices_train = {}
    label_to_indices_test = {}

    # Create two dictionaries to link/associate each class with all its training examples
    for integer_index, c in enumerate(dataset.classes_total):
        label_to_indices_train[c] = np.argwhere(dataset.y_train[:, integer_index] > 0).reshape(-1)
        label_to_indices_test[c] = np.argwhere(dataset.y_test[:, integer_index] > 0).reshape(-1)

    # Select data based on passed parameter
    x = dataset.x_train if from_train else dataset.x_test
    label_to_indices = label_to_indices_train if from_train else label_to_indices_test

    current_label_short = current_label[3:5]
    if direct_index:
        current_example_index = current_label_index
        compared_example_index = compared_label_index
    else:
        # Get examples from the right label and based on the passed index
        try:
            current_example_index = label_to_indices.get(current_label)[current_label_index]
            compared_example_index = label_to_indices.get(compared_label)[compared_label_index]
        except IndexError:
            raise IndexError(
                'At least one configured index is out of range of the number of examples with configured label.')

    single_plot(current_label, compared_label,
                current_label_short, compared_label_short,
                current_example_index, compared_example_index,
                mark_relevant, reduce_to_predefined_relevant, plot_ts, plot_combined, plot_matrix,
                x, dataset.feature_names_all)


def plot_multiple_labels(current_labels, compared_label,
                         compared_label_short, from_train, mark_relevant, reduce_to_predefined_relevant,
                         plot_ts, plot_combined, plot_matrix, instances_per_label):
    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    label_to_indices_train = {}
    label_to_indices_test = {}

    # Create two dictionaries to link/associate each class with all its training examples
    for integer_index, c in enumerate(dataset.classes_total):
        label_to_indices_train[c] = np.argwhere(dataset.y_train[:, integer_index] > 0).reshape(-1)
        label_to_indices_test[c] = np.argwhere(dataset.y_test[:, integer_index] > 0).reshape(-1)

    # Select data based on passed parameter
    x = dataset.x_train if from_train else dataset.x_test
    label_to_indices = label_to_indices_train if from_train else label_to_indices_test

    np.random.seed(23)
    indices_compared_label_all = np.random.choice(label_to_indices.get(compared_label), instances_per_label,
                                                  replace=False)

    for current_label in current_labels:
        current_label_short = current_label[3:5]

        indices_current_label = label_to_indices.get(current_label)
        if instances_per_label > len(indices_current_label):
            indices_compared_label = indices_compared_label_all[0:len(indices_current_label)]
        else:
            # Must be reassigned to other variable because otherwise we couldn't reduce it in case of less examples
            indices_compared_label = indices_compared_label_all
            indices_current_label = np.random.choice(label_to_indices.get(current_label), instances_per_label,
                                                     replace=False)

        for current_example_index, compared_example_index in zip(indices_current_label, indices_compared_label):
            single_plot(current_label, compared_label,
                        current_label_short, compared_label_short,
                        current_example_index, compared_example_index,
                        mark_relevant, reduce_to_predefined_relevant, plot_ts, plot_combined, plot_matrix, x,
                        dataset.feature_names_all)


def main():
    current_labels = ['txt15_conveyor_failure_mode_driveshaft_slippage_failure',
                      'txt15_i1_lightbarrier_failure_mode_1',
                      'txt15_i1_lightbarrier_failure_mode_2',
                      'txt15_i3_lightbarrier_failure_mode_2',
                      'txt15_m1_t1_high_wear',
                      'txt15_m1_t1_low_wear',
                      'txt15_m1_t2_wear',
                      'txt15_pneumatic_leakage_failure_mode_1',
                      'txt15_pneumatic_leakage_failure_mode_2',
                      'txt15_pneumatic_leakage_failure_mode_3',
                      'txt16_conveyor_failure_mode_driveshaft_slippage_failure',
                      'txt16_conveyorbelt_big_gear_tooth_broken_failure',
                      'txt16_conveyorbelt_small_gear_tooth_broken_failure',
                      'txt16_i3_switch_failure_mode_2',
                      'txt16_i4_lightbarrier_failure_mode_1',
                      'txt16_m3_t1_high_wear',
                      'txt16_m3_t1_low_wear',
                      'txt16_m3_t2_wear',
                      'txt16_turntable_big_gear_tooth_broken_failure',
                      'txt17_i1_switch_failure_mode_1',
                      'txt17_i1_switch_failure_mode_2',
                      'txt17_pneumatic_leakage_failure_mode_1',
                      'txt17_pneumatic_leakage_failure_mode_1_faulty',
                      'txt17_workingstation_transport_failure_mode_wout_workpiece',
                      'txt18_pneumatic_leakage_failure_mode_1',
                      'txt18_pneumatic_leakage_failure_mode_2',
                      'txt18_pneumatic_leakage_failure_mode_2_faulty',
                      'txt18_transport_failure_mode_wout_workpiece',
                      'txt19_i4_lightbarrier_failure_mode_1',
                      'txt19_i4_lightbarrier_failure_mode_2']

    batch_mode = True

    # Overall settings for batch mode
    compared_label = 'no_failure'
    compared_label_short = 'nf'
    from_train = False
    mark_relevant = True
    reduce_to_predefined_relevant = False
    plot_ts = True
    plot_combined = True
    plot_matrix = False
    instances_per_label = 5

    # Additional settings for single plot mode
    current_label = current_labels[2]
    current_label_index = 9
    compared_label_index = 69
    direct_index = False

    if batch_mode:
        starting_time = datetime.now()

        plot_multiple_labels(current_labels, compared_label, compared_label_short, from_train, mark_relevant,
                             reduce_to_predefined_relevant, plot_ts, plot_combined, plot_matrix, instances_per_label)

        end_time = datetime.now()

        print('Started @', starting_time.strftime("%H:%M:%S"))
        print('Finished @', end_time.strftime("%H:%M:%S"))

    else:
        plot_single_label(current_label, compared_label,
                          current_label_index, compared_label_index,
                          compared_label_short, from_train, mark_relevant,
                          reduce_to_predefined_relevant, plot_ts, plot_combined, plot_matrix, direct_index)


if __name__ == '__main__':
    main()
