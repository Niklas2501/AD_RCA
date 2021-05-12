import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from itertools import cycle, islice
from matplotlib import pyplot as plt
from configuration.Configuration import Configuration
from framework.Dataset import Dataset


def main():
    labels = ["txt16_conveyor_failure_mode_driveshaft_slippage_failure",
              "txt16_conveyorbelt_big_gear_tooth_broken_failure"]

    current_label = labels[0]
    compared_label = 'no_failure'
    current_label_index = 0
    compared_label_index = 0
    current_label_short = 'F'
    compared_label_short = 'NF'

    symptoms_current_example = {"a_16_3_x": "h",
                                "a_16_3_y": "h",
                                # "a_16_3_z": "h",
                                "txt16_m3.finished": "h",
                                "txt15_i8": "l"
                                }
    additional_features = ["a_15_1_x",
                           # "a_15_1_y",
                           # "a_15_1_z",
                           "txt15_m1.finished", ]
    displayed_features = list(symptoms_current_example.keys()) + additional_features

    config = Configuration()

    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    label_to_indices = {}

    for integer_index, c in enumerate(dataset.classes_total):
        label_to_indices[c] = np.argwhere(dataset.y_test[:, integer_index] > 0).reshape(-1)

    features = dataset.feature_names_all
    dataset = dataset.x_test

    try:
        current_example_index = label_to_indices.get(current_label)[current_label_index]
        compared_example_index = label_to_indices.get(compared_label)[compared_label_index]
    except IndexError:
        raise IndexError(
            'At least one configured index is out of range of the number of examples with configured label.')

    current_example = dataset[current_example_index]
    compared_example = dataset[compared_example_index]

    relevant_indices = np.isin(features, displayed_features)
    current_example = current_example[:, relevant_indices]
    compared_example = compared_example[:, relevant_indices]
    features = features[relevant_indices]

    renamed = []

    for feature in features:
        if feature in symptoms_current_example.keys():
            new = feature + " - " + symptoms_current_example.get(feature).upper()
        else:
            new = feature + " - " + "N"
        renamed.append(new)

    current_example_df = pd.DataFrame(data=current_example, columns=features)
    compared_example_df = pd.DataFrame(data=compared_example, columns=features)

    description = ''.join(['example-', current_label, '-', str(current_example_index), '-',
                           compared_label, '-', str(compared_example_index), ])
    file_name = ''.join(
        ['../data/visualization/', description])

    print('Creating plot with comparison for', description)

    # Append short label code to feature names and update in dataframes
    current_features = np.array([feature + ' - ' + current_label_short for feature in features])
    compared_features = np.array([feature + ' - ' + compared_label_short for feature in features])
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

    plot_simple(combined, file_name + '-combined', colors)


def plot_simple(df: pd.DataFrame, file_name, colors=None):
    plt.style.use("bmh")

    # Ensure subplot are distributed over the full plot, don't change the values
    plt.subplots_adjust(top=0.85, bottom=0.05)

    fig_width = 10
    fig_height = 15
    format = 'svg'

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

    # plt.suptitle('All sensors', fontsize=35, y=0.90)

    plt.savefig(file_name, dpi=300, bbox_inches="tight", format=format)


if __name__ == '__main__':
    main()
