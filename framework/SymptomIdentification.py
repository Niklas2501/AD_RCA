import jenkspy
import numpy as np


class SymptomIdentification:

    @staticmethod
    def symptom_identification(feature_scores, feature_names, mode, ss_percentiles_train, configured_percentile):

        if mode == 'sort_only':
            feature_names, _ = SymptomIdentification.sort_by(feature_names, feature_scores, feature_scores)
            return feature_names

        elif mode == 'diff_elbow':
            return SymptomIdentification.score_diff_elbow(feature_scores, feature_names)

        elif mode == 'rsm_gan_elbow':
            return SymptomIdentification.rsm_gan_elbow(feature_scores, feature_names)

        elif mode == 'train_percentile':
            return SymptomIdentification.train_percentile(feature_scores, feature_names,
                                                          ss_percentiles_train, configured_percentile)
        elif mode == 'example_percentile':
            return SymptomIdentification.example_percentile(feature_scores, feature_names,
                                                            configured_percentile)

        elif mode == 'jenks_breaks':
            return SymptomIdentification.jenks_breaks(feature_scores, feature_names)

        elif mode == 'jenks_percentile':
            return SymptomIdentification.jenks_percentile(feature_scores, feature_names,
                                                          ss_percentiles_train, configured_percentile)
        else:
            raise ValueError('Symptom selection mode not implemented: ', mode)

    @staticmethod
    def reduce_to(feature_names, feature_scores, reduce_to):
        feature_names = feature_names[reduce_to]
        feature_scores = feature_scores[reduce_to]

        return feature_names, feature_scores

    @staticmethod
    def sort_by(feature_names, feature_scores, sort_by):
        indices_sorted = sort_by.argsort()[::-1]
        names_sorted = feature_names[indices_sorted]
        scores_sorted = feature_scores[indices_sorted]
        return names_sorted, scores_sorted

    @staticmethod
    def score_diff_elbow(feature_scores, feature_names):
        """
        Returns the features that correspond to the n highest scores, for which holds that
        the difference between score[n] and score[n+1] is the highest

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :return: A np.ndarray with the names of the selected symptoms
        """
        names_sorted, scores_sorted = SymptomIdentification.sort_by(feature_names, feature_scores, feature_scores)

        diffs = scores_sorted[1:] - scores_sorted[0:-1]
        selected_index = np.argmin(diffs)

        # +1 so the selected index is included
        return names_sorted[0:selected_index + 1]

    @staticmethod
    def rsm_gan_elbow(feature_scores, feature_names):
        """
        Symptom selection as proposed by Khoshnevisan and Fan (https://arxiv.org/pdf/1911.07104.pdf)

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :return: A np.ndarray with the names of the selected symptoms
        """
        names_sorted, scores_sorted = SymptomIdentification.sort_by(feature_names, feature_scores, feature_scores)

        x = np.arange(0, scores_sorted.shape[0])
        x1, x2, y1, y2 = x[0], x[-1], scores_sorted[0], scores_sorted[-1]

        m = (y2 - y1) / (x2 - x1)
        b = (x2 * y1 - x1 * y2) / (x2 - x1)
        linear_func = np.vectorize(lambda x: m * x + b)

        distances_of_scores = np.abs(linear_func(x) - scores_sorted)
        selected_index = np.argmax(distances_of_scores)

        # print(*zip(names_sorted, distances_of_scores), sep='\n')

        # Paper states greater than the selected points --> no + 1
        return names_sorted[0:selected_index]

    @staticmethod
    def train_percentile(feature_scores, feature_names, ss_percentiles_train, configured_percentile):
        """
        Selects the features for which holds that the corresponding feature scores are above the percentile values in
        the training data of passed percentile. Those are additionally ordered by the height of the deviation between
        the  feature scores and the percentile values

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :param ss_percentiles_train: A np.ndarray containing the percentile values for each feature for each percentile
            with shape (100, number of features)
        :param configured_percentile: The percentile that should be used
        :return: A np.ndarray with the names of the selected symptoms
        """

        percentile_vector = ss_percentiles_train[configured_percentile]

        score_diff = np.subtract(feature_scores, percentile_vector)
        scores_over_percentile = np.greater(score_diff, 0)

        symptom_scores = feature_scores[scores_over_percentile]
        symptom_names = feature_names[scores_over_percentile]
        relevant_score_diff = score_diff[scores_over_percentile]

        # sorting based on deviation from percentile
        symptom_names_sorted, _ = SymptomIdentification.sort_by(symptom_names, symptom_scores, relevant_score_diff)

        return symptom_names_sorted

    @staticmethod
    def example_percentile(feature_scores, feature_names, configured_percentile):
        """
        Returns the features whose values are above the passed percentile of all values in the example.

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :param configured_percentile: The percentile that should be used
        :return: A np.ndarray with the names of the selected symptoms
        """

        percentile_value = np.percentile(feature_scores, configured_percentile)
        scores_over_percentile = np.greater(feature_scores, percentile_value)

        feature_names, feature_scores = SymptomIdentification.reduce_to(feature_names, feature_scores,
                                                                        scores_over_percentile)

        feature_names, feature_scores = SymptomIdentification.sort_by(feature_names, feature_scores, feature_scores)

        return feature_names

    @staticmethod
    def jenks_breaks(feature_scores, feature_names):
        """
        Selections of symptoms by applying the jenks natural breaks algorithm to the feature scores and returning
        the features that are assigned to the cluster (of overall clusters) which contains the higher values.

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :return: A np.ndarray with the names of the selected symptoms
        """

        res = jenkspy.jenks_breaks(feature_scores, nb_class=2)

        # Get the exclusive lower bound of the cluster with the highest scores
        lower_bound_exclusive = res[-2]

        # Boolean mask which features belong to this cluster
        is_symptom = np.greater(feature_scores, lower_bound_exclusive)

        # Reduce to those features
        feature_names, feature_scores = SymptomIdentification.reduce_to(feature_names, feature_scores, is_symptom)

        return feature_names

    @staticmethod
    def jenks_percentile(feature_scores, feature_names, ss_percentiles_train, configured_percentile):
        """
        Selection of symptoms by calculation of the deviation between the feature scores and the percentile values of
        the features in the training dataset for the configured_percentile and splitting those deviation scores
        into two clusters using the jenks natural breaks algorithm.

        :param feature_names: A np.ndarray with the names that correspond the scores
        :param feature_scores: A np.ndarray with the scores based on which the names should be selected
        :param ss_percentiles_train: A np.ndarray containing the percentile values for each feature for each percentile
            with shape (100, number of features)
        :param configured_percentile: The percentile that should be used
        :return: A np.ndarray with the names of the selected symptoms
        """

        percentile_vector = ss_percentiles_train[configured_percentile]
        score_diff = np.subtract(feature_scores, percentile_vector)

        res = jenkspy.jenks_breaks(score_diff, nb_class=2)

        # Get the exclusive lower bound of the cluster with the highest difference between scores and percentile
        lower_bound_exclusive = res[-2]

        # Boolean mask which features belong to this cluster
        is_symptom = np.greater(score_diff, lower_bound_exclusive)

        # Reduce to those features
        feature_names, feature_scores = SymptomIdentification.reduce_to(feature_names, feature_scores, is_symptom)

        return feature_names
