import os
import sys

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from framework.Ontology import SemanticModel
from configuration.Configuration import Configuration


def bar_plot():
    features = [
        'a_15_1_x', 'a_15_1_y', 'a_15_1_z', 'a_16_3_x', 'a_16_3_y', 'a_16_3_z', 'hPa_15', 'hPa_17', 'hPa_18',
        'txt15_i1',
        'txt15_i2', 'txt15_i3', 'txt15_i6', 'txt15_i7', 'txt15_i8', 'txt15_m1.finished', 'txt15_o5', 'txt15_o6',
        'txt15_o7',
        'txt15_o8', 'txt16_i1', 'txt16_i2', 'txt16_i3', 'txt16_i4', 'txt16_i5', 'txt16_m1.finished',
        'txt16_m2.finished',
        'txt16_m3.finished', 'txt16_o7', 'txt16_o8', 'txt17_i1', 'txt17_i2', 'txt17_i3', 'txt17_i5',
        'txt17_m1.finished',
        'txt17_m2.finished', 'txt17_o5', 'txt17_o6', 'txt17_o7', 'txt17_o8', 'txt18_i1', 'txt18_i2', 'txt18_i3',
        'txt18_m1.finished', 'txt18_m2.finished', 'txt18_m3.finished', 'txt18_o7', 'txt18_o8', 'txt18_vsg_x',
        'txt18_vsg_y',
        'txt18_vsg_z', 'txt19_i1', 'txt19_i4', 'txt19_i5', 'txt19_i6', 'txt19_i7', 'txt19_i8', 'txt19_m1.finished',
        'txt19_m2.finished', 'txt19_m3.finished', 'txt19_m4.finished',
    ]
    features = np.array(features)

    config = Configuration()
    dict = config.component_symptom_selection

    counts_low = np.zeros(shape=(61,))
    counts_med = np.zeros(shape=(61,))
    counts_high = np.zeros(shape=(61,))

    for component, symptoms in dict.items():
        high, medium, low, unlikely = symptoms
        all = high + medium + low + unlikely

        for symptom in all:

            feature_index = np.where(features == symptom)

            if symptom in high:
                counts_high[feature_index] += 1
            elif symptom in medium:
                counts_med[feature_index] += 1
            elif symptom in low:
                counts_low[feature_index] += 1

    # noinspection PyTypeChecker
    df = pd.DataFrame([counts_high, counts_med, counts_low], columns=features,
                      index=['high', 'medium', 'low'])
    # df = df.drop(index=['low'])

    # noinspection PyUnresolvedReferences
    only_0_cols = df.loc[:, ~(df != 0).any(axis=0)].columns
    df = df.drop(columns=only_0_cols)

    print('Fully irrelevant features:', len(only_0_cols))
    print(only_0_cols)

    f_names = []
    for i in range(len(df.columns)):
        f_names.append('f' + str(i))
    f_names = np.array(f_names)

    df = df.T
    df['f_names'] = f_names
    df['sum'] = df.sum(axis=1)
    df = df.sort_values(by='sum', ascending=False)
    print(df.to_string())
    f_names = list(df['f_names'])
    df = df.drop(columns=['sum', 'f_names'])

    ax = df.plot.bar(stacked=True, figsize=(12, 8), width=0.5)
    ax.set_xlabel("Features")
    ax.set_ylabel("Number of components")
    plt.xticks(ticks=np.arange(0, len(f_names)), labels=f_names, rotation=0)
    plt.tight_layout()
    plt.show()


def heatmap():
    drop_0 = True
    sort_by_sum = False
    relevance_labels = ['Irrelevant', 'Low', 'Medium', 'High']
    cmap = colors.ListedColormap(['#545454', '#0088ff', '#0aad36', '#fff200'])
    mapping = {
        "h": 1.0,
        "m": 0.5,
        "l": 0.3,
        "e": 0.0
    }

    config = Configuration()
    ontology = SemanticModel(config)
    ontology.import_from_file()
    ontology.update_scores(mapping)

    components, relevance_vectors, f_names = [], [], []

    for comp, vec in ontology.relevance_knowledge_scores.items():
        components.append(comp)
        relevance_vectors.append(vec)

    relevance_vectors = np.vstack(relevance_vectors)

    for i in range(relevance_vectors.shape[1]):
        f_names.append('f' + str(i))
    f_names = np.array(f_names)

    import pandas as pd

    df = pd.DataFrame(relevance_vectors.T, columns=components, index=f_names)
    df['sort_by'] = df.sum(axis=1) if sort_by_sum else np.count_nonzero(df, axis=1)
    df = df.sort_values(by='sort_by', ascending=False)

    if drop_0:
        print('Dropping {} irrelevant features'.format(len(df[df['sort_by'] == 0])))
        df = df[df['sort_by'] != 0]

    df = df.drop(columns=['sort_by'])

    relevance_vectors = df.values.T
    f_names = df.index.values

    font = {'family': 'normal',
            'size': 26}

    import matplotlib
    matplotlib.rc('font', **font)

    plt.figure(figsize=(31, 10))

    ax = plt.gca()
    im = ax.imshow(relevance_vectors, cmap=cmap, aspect='auto')

    ax.set_yticks(np.arange(relevance_vectors.shape[0]))
    ax.set_yticklabels(components)

    ax.set_xticks(np.arange(relevance_vectors.shape[1]))
    ax.set_xticklabels(f_names, )

    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', fraction=0.07, pad=0.1)
    cbar.ax.set_xlabel('Relevance')

    tick_locs = np.array([0.125, 3 * 0.125, 5 * 0.125, 7 * 0.125])
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(relevance_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(relevance_vectors.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(relevance_vectors.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    # plt.savefig('plot.png')
    plt.show()


def main():
    heatmap()


if __name__ == '__main__':
    main()
