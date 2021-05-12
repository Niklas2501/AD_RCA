import json
import sys

import pandas as pd

from configuration.Configuration import Configuration


def print_comp_feature_comb(df, class_component_grouping, class_feature_grouping):
    for component in class_component_grouping.keys():
        print(component)

        df_comp = pd.DataFrame(None, columns=df.columns)

        for label in class_component_grouping.get(component):
            df_comp.loc[label] = ''

            selection = class_feature_grouping.get(label)
            df_comp.loc[label, selection[0]] = 'h'
            df_comp.loc[label, selection[1]] = 'm'
            df_comp.loc[label, selection[2]] = 'l'
            df_comp.loc[label, selection[3]] = 'l-q'

        df = df.sort_index(axis=1)
        if df_comp.empty:
            print('There are no classes associated with this component, check the classToComponents table.')
        else:
            print(df_comp.to_string())
        print('\n')


def df_to_dict(df: pd.DataFrame):
    return_dict = {}

    for label, row in df.iterrows():
        high = row.index[row == 'h'].to_list()
        med = row.index[row == 'm'].to_list()
        low = row.index[row == 'l'].to_list()
        questionable = row.index[row == 'l-q'].to_list()

        if len(high) + len(med) + len(low) == 0:
            print(label, 'has no relevant attributes and should be removed.')

        return_dict[label] = [high, med, low, questionable]

    return return_dict


def export(class_component_grouping, class_feature_grouping, component_feature_grouping):
    pass


def main():
    # If true the mapping will be read from the component relevance table and exported into a json dictionary
    # If false for each component a df will be printed which contains the associated features and their feature mapping
    export_component_features = True

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    config = Configuration()

    file = config.data_folder_prefix + 'knowledge/feature_selection.xlsx'
    df = pd.read_excel(file, sheet_name=1, index_col='Component \ Class')
    component_to_class = {}

    for component, row in df.iterrows():
        component_to_class[component] = row.index[row == 1.0].to_list()

    df = pd.read_excel(file, sheet_name=0, index_col='Classes \ Features')

    class_symptom_selection = df_to_dict(df)

    print()

    if not export_component_features:
        print_comp_feature_comb(df, component_to_class, class_symptom_selection)
        sys.exit(0)

    df = pd.read_excel(file, sheet_name=2, index_col='Component \ Features')
    component_symptom_selection = df_to_dict(df)

    # for key in component_feature_grouping.keys():
    #     print(key, component_feature_grouping.get(key))

    json_file_content = {
        'component_to_class': component_to_class,
        # 'class_symptom_selection': class_symptom_selection,
        'component_symptom_selection': component_symptom_selection
    }

    print(json.dumps(json_file_content, indent=2))


if __name__ == '__main__':
    main()
