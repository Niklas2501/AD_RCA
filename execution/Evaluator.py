from math import floor

import numpy as np
import pandas as pd
from sklearn.metrics import *

from configuration.Configuration import Configuration
from framework.Dataset import Dataset


class Evaluator:

    def __init__(self, config, dataset, num_test_examples=-1):
        self.config: Configuration = config
        self.dataset: Dataset = dataset

        if self.config.test_with_validation_dataset:
            self.y_strings = self.dataset.y_test_val_strings
            self.y_anomaly_mask = self.dataset.y_test_val_anomaly_mask
            self.num_examples = self.dataset.num_test_val_instances
        else:
            self.y_strings = self.dataset.y_test_strings
            self.y_anomaly_mask = self.dataset.y_test_anomaly_mask
            self.num_examples = self.dataset.num_test_instances

        if num_test_examples > 0:
            self.num_examples = num_test_examples

        self.f_score_beta_str = 'F' + str(self.config.f_score_beta)

        # Dataframe that stores the results that will be output at the end of the inference process
        # Is not filled with data during the inference
        index = list(set(self.y_strings)) + ['combined']
        cols = ['#Examples', 'TP', 'FP', 'TN', 'FN', 'ACC', 'FNR', 'TNR', 'FPR', 'TPR', 'Prec', 'F1']
        ad_specific_cols = [self.f_score_beta_str, 'AVG # affected']

        self.ad_results_class_based = pd.DataFrame(0, index=index, columns=cols + ad_specific_cols)
        self.ad_results_class_based.index.name = 'Classes'
        self.ad_results_class_based.loc['combined', '#Examples'] = self.num_examples

        index = ['no_failure'] + list(self.config.component_to_class.keys()) + ['combined']
        self.ad_results_intermed = pd.DataFrame(0, index=index, columns=cols + ad_specific_cols)
        self.ad_results_intermed.index.name = 'Component'
        self.ad_results_intermed.loc['combined', '#Examples'] = self.num_examples

        self.ad_results_final = self.ad_results_intermed.copy(deep=True)

        cols = ['#Examples', 'TP', 'FP', 'TN', 'FN', 'TPR', 'Prec', 'F1'] + ['AVG-HR@K', 'AVG-HR@100%', 'AVG-HR@150%']
        index = list(self.config.component_to_class.keys()) + ['none', 'combined']
        self.si_st_results = pd.DataFrame(0, index=index, columns=cols)
        self.si_st_results.index.name = 'Component'

        self.component_gt = []
        self.component_pred = []

        # Prepare ground truth for symptom identification evaluation
        self.symptom_gt = {}
        for component in self.config.component_symptom_selection.keys():
            all_symptoms = self.config.component_symptom_selection.get(component)
            high, medium, low = all_symptoms[0], all_symptoms[1], all_symptoms[2]
            gt_for_component = np.array(high + medium)

            self.symptom_gt[component] = gt_for_component

    def add_ad_result(self, initial_prediction: bool, a_t_vector, example_index: int, final_prediction: bool):

        """
        :param initial_prediction Anomaly decision before the FPD
        :param a_t_vector A np.ndarray that stores which time stamp were predicted to be abnormal
        :param example_index: The index of the evaluated example in the dataset
        :param final_prediction Anomaly decision after the FPD
        """

        # Get the true label stored in the dataset
        true_class = self.y_strings[example_index]
        true_component = true_class if true_class == 'no_failure' else self.config.class_to_component.get(true_class)
        nbr_affected_ts = np.count_nonzero(a_t_vector)
        is_anomaly = self.y_anomaly_mask[example_index]  # Ground truth anomaly decision

        self.ad_results_class_based = self.add_prediction(self.ad_results_class_based, true_class, nbr_affected_ts,
                                                          is_anomaly, initial_prediction)

        self.ad_results_intermed = self.add_prediction(self.ad_results_intermed, true_component,
                                                       nbr_affected_ts, is_anomaly, initial_prediction)

        self.ad_results_final = self.add_prediction(self.ad_results_final, true_component,
                                                    nbr_affected_ts, is_anomaly, final_prediction)

        nbr_tested = self.get_nbr_examples_tested()
        current_correct = self.get_nbr_correct()

        if self.config.enable_intermediate_output:

            # create output for this example
            example_results = [
                ['Example:', str(nbr_tested) + '/' + str(self.num_examples)],
                ['Contains anomaly?', str(is_anomaly)],
                ['Initial prediction:', str(initial_prediction)],
                ['Final prediction:', str(final_prediction)],
                ['Affected time stamps:', str(nbr_affected_ts)],
                ['Correctly predicted:', str(current_correct) + '/' + str(nbr_tested)],
                ['Correctly predicted %:', (current_correct / nbr_tested) * 100.0],
            ]

            # output results for this example
            for row in example_results:
                print("{: <30} {: <30}".format(*row))

    @staticmethod
    def add_prediction(target_df: pd.DataFrame, true_label: str, nbr_affected_ts: int, is_anomaly: bool,
                       is_predicted_as_anomaly: bool):
        """

        :param target_df: The dataframe the prediction should be added to
        :param true_label: The ground truth label of the tested example
        :param nbr_affected_ts: The number of affected timestamps calculated by the framework
        :param is_anomaly: The ground truth anomaly decision of the tested example
        :param is_predicted_as_anomaly: The anomaly decision of the framework
        :return:
        """
        target_df.loc[true_label, '#Examples'] += 1
        target_df.loc[true_label, 'AVG # affected'] += nbr_affected_ts

        if is_anomaly and is_predicted_as_anomaly:
            target_df.loc[true_label, 'TP'] += 1
        elif is_anomaly and not is_predicted_as_anomaly:
            target_df.loc[true_label, 'FN'] += 1
        elif not is_anomaly and is_predicted_as_anomaly:
            target_df.loc[true_label, 'FP'] += 1
        elif not is_anomaly and not is_predicted_as_anomaly:
            target_df.loc[true_label, 'TN'] += 1

        return target_df

    def add_si_st_result(self, selected_symptoms, st_components, st_scores, example_index: int):
        """
        :param selected_symptoms: The symptoms selected by the SI for the example
        :param st_components: The labels of the components as ranked by the ST module
        :param st_scores: The scores of the components as determined by the ST module
        :param example_index: The index of the evaluated example in the dataset
        """
        true_class = self.y_strings[example_index]

        if true_class == 'no_failure':
            if self.config.enable_intermediate_output:
                print()

            return

        # Get predicted component of the ST module, i.e. the one with the highest rank
        predicted_component, score = st_components[0], st_scores[0]
        true_component = self.config.class_to_component.get(true_class)
        k_components = st_components[0:self.config.k_for_hits_at_k]

        if true_component in k_components:
            self.si_st_results.loc[true_component, 'AVG-HR@K'] += 1

        # Fill lists with predicted and ground truth labels in order to
        # calculate the multiclass confusion matrix at the end
        self.component_gt.append(true_component)
        self.component_pred.append(predicted_component)

        symptoms_gt = self.symptom_gt.get(true_component)
        symptoms_gt_string = ", ".join(symptoms_gt)

        nbr_displayed_symptoms = int(len(symptoms_gt) * 1.6)
        nbr_displayed_symptoms = nbr_displayed_symptoms if nbr_displayed_symptoms < len(selected_symptoms) else len(
            selected_symptoms)

        symptoms_string = ", ".join(selected_symptoms[0:nbr_displayed_symptoms])
        k_components_string = ", ".join(k_components)

        hr100 = self.hit_rate_at_p(selected_symptoms, symptoms_gt, 100)
        hr150 = self.hit_rate_at_p(selected_symptoms, symptoms_gt, 150)

        self.si_st_results.loc[true_component, 'AVG-HR@100%'] += hr100
        self.si_st_results.loc[true_component, 'AVG-HR@150%'] += hr150

        if self.config.enable_intermediate_output:

            # create output for this example
            example_results = [
                ['GT component:', true_component],
                ['k components:', k_components_string],
                ['Component score:', score],
                ['GT symptoms:', symptoms_gt_string],
                ['Predicted symptoms:', symptoms_string],
                ['HitRate@100%', hr100],
                ['HitRate@150%', hr150]
            ]

            # output results for this example
            for row in example_results:
                print("{: <30} {: <30}".format(*row))
            print()

    @staticmethod
    def hit_rate_at_p(predicted, ground_truth, percent):
        """
        :param predicted: The symptoms predicted to be relevant by the SI
        :param ground_truth: The ground truth of relevant attributes
        :param percent: Which variant should be calculated.
        :return: Hit Rate at percent %
        """

        assert percent in [100, 150], 'Uncommon percentage used. Define in assertion or change.'

        len_gt = float(len(ground_truth))

        # Calculate the # predictions that are compared to the ground truth based on its length and the percentage
        len_gt_weighted = floor(len_gt * (percent / 100.0))

        # Check case that 1,5 * gt could be more than overall number of features
        len_gt_weighted = len_gt_weighted if len_gt_weighted <= len(predicted) else len(predicted)

        # Reduce the predictions to the calculated number
        predicted_relevant_selection = predicted[0:len_gt_weighted]

        # Sum up how many predictions are in the ground truth
        hit_rate = 0
        for symptom in predicted_relevant_selection:
            hit_rate += 1 if symptom in ground_truth else 0

        return hit_rate / len_gt

    def calculate_results(self):
        """
        Method for executing all calculations needed before the overall evaluation results can be output
        """

        self.ad_results_class_based = self.calculate_generic_results(self.ad_results_class_based)
        self.ad_results_class_based = self.calculate_ad_results(self.ad_results_class_based, self.config.f_score_beta,
                                                                self.f_score_beta_str)

        self.ad_results_intermed = self.calculate_generic_results(self.ad_results_intermed)
        self.ad_results_intermed = self.calculate_ad_results(self.ad_results_intermed, self.config.f_score_beta,
                                                             self.f_score_beta_str)

        self.ad_results_final = self.calculate_generic_results(self.ad_results_final)
        self.ad_results_final = self.calculate_ad_results(self.ad_results_final, self.config.f_score_beta,
                                                          self.f_score_beta_str)

        # Important that the generic function is called afterwards because the specific one has to add
        # relevant values first
        self.si_st_results = self.calculate_si_st_results(self.si_st_results, self.component_gt, self.component_pred)

    @staticmethod
    def calculate_generic_results(result_df: pd.DataFrame):
        """
        Compute general metrics
        :param result_df: A dataframe containing results for any kind of evaluation (AD  or SI/ST)
        :return: The input dataframe with additional metrics calculated
        """

        # Fill the combined row with the sum of each class
        result_df.loc['combined', 'TP'] = result_df['TP'].sum()
        result_df.loc['combined', 'TN'] = result_df['TN'].sum()
        result_df.loc['combined', 'FP'] = result_df['FP'].sum()
        result_df.loc['combined', 'FN'] = result_df['FN'].sum()

        # Calculate the classification accuracy for all classes and save in the intended column
        # Don't divide by example column because this  wouldn't work in the multi label case
        result_df['ACC'] = (result_df['TP'] + result_df['TN']) / result_df[['TP', 'TN', 'FP', 'FN']].sum(
            axis=1).replace(0, np.NaN)

        result_df['TPR'] = Evaluator.rate_calculation(result_df['TP'], result_df['FN'])
        result_df['FPR'] = Evaluator.rate_calculation(result_df['FP'], result_df['TN'])
        result_df['TNR'] = Evaluator.rate_calculation(result_df['TN'], result_df['FP'])
        result_df['FNR'] = Evaluator.rate_calculation(result_df['FN'], result_df['TP'])

        result_df['Prec'] = Evaluator.rate_calculation(result_df['TP'], result_df['FP'])
        result_df['F1'] = 2 * (result_df['TPR'] * result_df['Prec']) / (
                result_df['TPR'] + result_df['Prec'])

        return result_df

    @staticmethod
    def rate_calculation(numerator, denominator_part2):
        """
        Used to calculate various metrics that follow the same pattern

        :param numerator: The numerator of the fraction
        :param denominator_part2: The second part of the denominator, summed with the numerator
        :return: numerator / (numerator + denominator_part2)
        """

        return numerator / (numerator + denominator_part2).replace(0, np.NaN)

    @staticmethod
    def calculate_ad_results(result_df: pd.DataFrame, f_score_beta: int, f_score_beta_col: str):
        """
        :param result_df: A dataframe containing results of the anomaly detection
        :param f_score_beta: The value of parameter beta for the F score calculation
        :param f_score_beta_col: The name of the column that contains the F score as string.
        :return: The input dataframe with additional values calculated
        """

        result_df.loc['combined', 'AVG # affected'] = result_df['AVG # affected'].sum()
        result_df['AVG # affected'] = result_df['AVG # affected'] / result_df['#Examples'].replace(0, np.NaN)

        if f_score_beta_col in result_df.columns:
            result_df[f_score_beta_col] = (1 + f_score_beta ** 2) * (result_df['TPR'] * result_df['Prec']) / (
                    result_df['TPR'] + (f_score_beta ** 2) * result_df['Prec'])

        return result_df

    @staticmethod
    def calculate_si_st_results(result_df, component_gt, component_pred):
        """
        :param result_df: A dataframe containing results of the SI / ST
        :param component_gt: The ground truth of  affected components for all evaluated (failure) examples
        :param component_pred: The predicted components to be affected for all evaluated (failure) examples
        :return: The input dataframe with additional values calculated
        """

        mcm = multilabel_confusion_matrix(y_true=component_gt, y_pred=component_pred)

        # Compile a list of components in the order that is returned by the mcm method
        components_in_conf = np.unique(np.array(component_gt + component_pred))
        components_in_conf.sort()

        # Assign tp, etc. for all components
        tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]

        # Combine tp, etc. in a dataframe with the same structure as the overall result dataframe
        cols = ['TP', 'FP', 'TN', 'FN']
        combined = [components_in_conf, tp, fp, tn, fn]
        temp_df = pd.DataFrame(combined, index=['Component'] + cols).T
        temp_df = temp_df.set_index('Component')

        # Add the calculated results into the overall result dataframe and calculate number of example from those
        result_df.loc[components_in_conf, cols] = temp_df[cols]
        result_df.loc[components_in_conf, '#Examples'] = temp_df[['TP', 'FN']].sum(axis=1).astype(int)
        result_df.loc['combined', '#Examples'] = result_df['#Examples'].sum(axis=0).astype(int)

        # Fill the combined row with the sum of each class
        result_df.loc['combined', 'TP'] = result_df['TP'].sum()
        result_df.loc['combined', 'TN'] = result_df['TN'].sum()
        result_df.loc['combined', 'FP'] = result_df['FP'].sum()
        result_df.loc['combined', 'FN'] = result_df['FN'].sum()

        if 'AVG-HR@K' in result_df.columns:
            # The column contains the sum of the @K Hits, so it needs to be divided by the number of examples
            result_df['AVG-HR@K'] = result_df['AVG-HR@K'] / result_df['#Examples'].replace(0, np.NaN)
            wout_comb = result_df.drop(['combined', 'none'], axis=0)

            # Calculate the combined values as weighted average
            result_df.loc['combined', 'AVG-HR@K'] = np.average(wout_comb['AVG-HR@K'],
                                                               weights=wout_comb['#Examples'])

        if 'AVG-HR@100%' in result_df.columns:
            result_df['AVG-HR@100%'] = result_df['AVG-HR@100%'] / result_df['#Examples'].replace(0, np.NaN)
            result_df['AVG-HR@150%'] = result_df['AVG-HR@150%'] / result_df['#Examples'].replace(0, np.NaN)
            wout_comb = result_df.drop(['combined', 'none'], axis=0)

            result_df.loc['combined', 'AVG-HR@100%'] = np.average(wout_comb['AVG-HR@100%'],
                                                                  weights=wout_comb['#Examples'])
            result_df.loc['combined', 'AVG-HR@150%'] = np.average(wout_comb['AVG-HR@150%'],
                                                                  weights=wout_comb['#Examples'])

        # Calculate metrics for each label
        result_df['TPR'] = Evaluator.rate_calculation(result_df['TP'], result_df['FN'])
        result_df['Prec'] = Evaluator.rate_calculation(result_df['TP'], result_df['FP'])
        result_df['F1'] = 2 * (result_df['TPR'] * result_df['Prec']) / (
                result_df['TPR'] + result_df['Prec'])

        # Correct metrics for combined row as weighted average over different labels based on the number of examples
        average = 'weighted'
        result_df.loc['combined', 'TPR'] = recall_score(component_gt, component_pred, average=average, zero_division=0)
        result_df.loc['combined', 'F1'] = f1_score(component_gt, component_pred, average=average, zero_division=0)
        result_df.loc['combined', 'Prec'] = precision_score(component_gt, component_pred, average=average,
                                                            zero_division=0)

        return result_df

    def st_reduced_to_txt(self):
        """
        Performs the additional evaluation of the ST module at workstation level.
        """

        # Reduce to txt labels
        component_pred = self.component_pred
        component_gt = self.component_gt
        component_pred = [comp[0:5] if comp != 'none' else 'none' for comp in component_pred]
        component_gt = [comp[0:5] if comp != 'none' else 'none' for comp in component_gt]

        # Get unique labels
        components_in_conf = np.unique(np.array(component_gt + component_pred))
        components_in_conf.sort()

        # Create dataframe to store the results
        cols = ['#Examples', 'TP', 'FP', 'TN', 'FN', 'TPR', 'Prec', 'F1']
        index = list(components_in_conf) + ['combined']
        st_txt_results = pd.DataFrame(0, index=index, columns=cols)
        st_txt_results.index.name = 'Component'

        # Apply the standard SI / ST calculation method
        st_txt_results = self.calculate_si_st_results(st_txt_results, component_gt=component_gt,
                                                      component_pred=component_pred)
        print(st_txt_results.to_string())

    def print_results(self):
        """
        Pretty print the results of the evaluation for the AD and SI / ST
        """

        spacer = 2 * '-----------------------------------------------------------------------------'
        metrics = [
            'TPR = True Positive Rate/Recall',
            'FPR = False Positive Rate'
            'TNR = True Negative Rate/Specificity',
            'FNR = False Negative Rate/Miss rate',
            'Prec = Precision',
            'ACC = Accuracy'
        ]

        print(spacer)
        print('Final Result for model ' + self.config.filename_model_to_use + ' :')
        print(spacer)
        print('Metrics:')
        print(*metrics, sep='\n')
        print(spacer)
        # print('Anomaly detection results per class:')
        # print(ad_results.to_string())
        # print(spacer)
        print('Anomaly detection results (based on INTERMEDIATE predictions): *1')
        print(spacer)
        print(self.ad_results_intermed.to_string())
        print(spacer)
        print('Anomaly detection results (based on FINAL predictions): *1')
        print(spacer)
        print(self.ad_results_final.to_string())
        print(spacer)
        print('*1 Note that these tables do NOT have the semantics of a multi class confusion matrix.'
              ' For each component a binary classification semantic ')
        print('(is anomaly or not) is displayed. i.e. there won\'t be any TP/FN for no_failure '
              'and no FP/TN for the individual components.')
        print(spacer)
        print('Symptom identification and tracing results per component: *2')
        print(spacer)
        print(self.si_st_results.to_string())
        print(spacer)
        print('Symptom tracing reduced to work station granularity: *2')
        print(spacer)
        self.st_reduced_to_txt()
        print(spacer)
        print(
            '*2 Note that the metrics in the combined row are calculated as '
            'weighted average of the scores of the individual classes')
        print('based on the number of examples.')
        print(spacer)

    def print_baseline_results(self, baseline_variant, parameters_used):
        spacer = 2 * '-----------------------------------------------------------------------------'
        metrics = [
            'TPR = True Positive Rate/Recall',
            'FPR = False Positive Rate'
            'TNR = True Negative Rate/Specificity',
            'FNR = False Negative Rate/Miss rate',
            'Prec = Precision',
            'ACC = Accuracy'
        ]

        print(spacer)
        print('Results for a {} model:'.format(baseline_variant))
        print('Parameters: {}'.format(parameters_used))
        print(spacer)
        print('Metrics:')
        print(*metrics, sep='\n')
        print(spacer)
        print('Anomaly detection results: *')
        print(self.ad_results_final.drop(columns=['AVG # affected']).to_string())
        print(spacer)
        print('* Note that these tables do NOT have the semantics of a multi class confusion matrix.'
              ' For each component a binary classification semantic (is anomaly or not) is displayed.')
        print('i.e. there won\'t be any TP/FN for no_failure and no FP/TN for the individual components.')
        print(spacer)

    def get_nbr_examples_tested(self):
        """
        :return: The number of examples already tested
        """

        return self.ad_results_intermed['#Examples'].drop('combined', axis=0).sum()

    def get_nbr_correct(self, from_corrected=True):
        """
        :param from_corrected Whether number of correct predictions
            should be based on the intermediate or fina prediction. Default: Based on final prediction
        :return: The number of examples predicted correctly up to this point
        """

        result_df = self.ad_results_final if from_corrected else self.ad_results_intermed

        return result_df['TP'].drop('combined', axis=0).sum() \
               + result_df['TN'].drop('combined', axis=0).sum()
