import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from owlready2 import *
from configuration.Configuration import Configuration
from framework.Dataset import Dataset


class SemanticModel:

    def __init__(self, config: Configuration, dataset: Dataset = None):
        self.config = config
        self.dataset = dataset
        self.ontology = get_ontology('http://iot.uni-trier.de/ADOnto')

        # Knowledge that can be retrieved from the ontology
        self.feature_names = None
        self.feature_name_to_index = {}
        self.relevance_knowledge_str = {}
        self.relevance_knowledge_scores = {}

    def get_iri(self, name: str):
        """
        :param name: The name of the individual the full iri is needed
        :return: The full iri of the individual with the passed name
        """
        with self.ontology:
            return self.ontology.base_iri + name

    def get_individual(self, individual_class, name: str):
        """
        :param individual_class: The python class of the the individual that should be searched for
        :param name: The name of the individual that should be searched for
        :return: The object instance of the search individual
        """

        with self.ontology:
            return self.ontology.search_one(type=individual_class, iri=self.get_iri(name))

    # noinspection PyUnusedLocal,PyPep8Naming
    def create(self, save_to_file: bool = False):
        """
        Creates the ontology from configured features, components and relevance scores.
        Saves the ontology to the configured file if enabled.
        :param save_to_file: Choice whether the save the created ontology to file or not.
        """
        assert dataset is not None, 'Dataset can not be None.'

        with self.ontology:

            # Define ontology classes and properties
            class Feature(Thing):
                pass

            class RelevanceScore(Thing):
                pass

            class Component(Thing):
                pass

            class has_index_in_data(DataProperty):
                domain = [Feature]
                range = [int]

            class has_score(DataProperty):
                domain = [RelevanceScore]
                range = [str]

            class of_component(ObjectProperty):
                domain = [RelevanceScore]
                range = [Component]

            class comp_has_rs(ObjectProperty):
                domain = [Component]
                range = [RelevanceScore]
                inverse_property = of_component

            class feature_has_rs(ObjectProperty):
                domain = [Feature]
                range = [RelevanceScore]

            class of_feature(ObjectProperty):
                domain = [RelevanceScore]
                range = [Feature]
                inverse_property = feature_has_rs

            class Sensor(Thing):
                pass

            class records(ObjectProperty):
                domain = [Sensor]
                range = [Feature]

            class recorded_by(ObjectProperty):
                domain = [Feature]
                range = [Sensor]

            # Component individuals
            txt15_i1 = Component(name='txt15_i1')
            txt15_i3 = Component(name='txt15_i3')
            txt15_conveyor = Component(name='txt15_conveyor')
            txt15_m1 = Component(name='txt15_m1')
            txt15_pl = Component(name='txt15_pl')
            txt16_i3 = Component(name='txt16_i3')
            txt16_conveyor = Component(name='txt16_conveyor')
            txt16_m3 = Component(name='txt16_m3')
            txt16_turntable = Component(name='txt16_turntable')
            txt17_i1 = Component(name='txt17_i1')
            txt17_pl = Component(name='txt17_pl')
            txt18_pl = Component(name='txt18_pl')
            txt19_i4 = Component(name='txt19_i4')

            # Create feature individuals and save location in dataset and the manually defined anomaly threshold
            for index_in_data, feature_name in enumerate(self.dataset.feature_names_all):
                feature = Feature(name=feature_name)
                feature.has_index_in_data = [index_in_data]

            # Link components and features to relevance scores
            for component in Component.instances():
                c_name = component.name

                high, medium, low, _ = self.config.component_symptom_selection.get(c_name)
                all_symptoms = high + medium + low

                # Create relevance score individuals for each defined dependency
                for i, symptom in enumerate(all_symptoms):
                    r_score_name = c_name + '_' + symptom + '_score'
                    r_score = RelevanceScore(name=r_score_name)

                    symptom_individual = self.get_individual(Feature, symptom)

                    # Important: Assigned scores must be 1 char strings and can not be 'e' (assigned to all others)
                    if symptom in high:
                        r_score.has_score = ['h']
                    elif symptom in medium:
                        r_score.has_score = ['m']
                    elif symptom in low:
                        r_score.has_score = ['l']

                    # Link relevance score individual to corresponding feature and component
                    # IMPORTANT: Don't assign inverse properties or this will break for whatever reason
                    r_score.of_feature = [symptom_individual]
                    r_score.of_component = [component]

        if save_to_file:
            self.ontology.save(file=self.config.ontology_file, format="rdfxml")

    def import_from_file(self):
        """
        Retrieves the knowledge needed from the ontology and stores it an an and stores it in directly applicable format
        """

        onto_path.append(self.config.ontology_file)

        self.ontology = get_ontology("file://" + self.config.ontology_file).load()

        with self.ontology:

            # Redefine relevant classes and relationships needed for retrieving the knowledge from the ontology
            class Feature(Thing):
                pass

            class RelevanceScore(Thing):
                pass

            class Component(Thing):
                pass

            # Retrieve features and store the names with the correct indices (matches self.dataset.feature_names_all,
            # which is not used in order to emphasize the ontology approach)
            features = []
            for feature in Feature.instances():
                f_name = feature.name
                f_index = feature.has_index_in_data[0]
                features.append((f_name, f_index))

            self.feature_names = np.empty(len(features), dtype=object)

            for f_name, f_index in features:
                self.feature_names[f_index] = f_name
                self.feature_name_to_index[f_name] = f_index

            # Retrieve component individuals and create a dictionary entry that stores the feature relevance vector for
            # this component. Note that the array is filled with the configured default
            # value of relevance when none is explicitly defined
            for component in Component.instances():
                relevance_vector = np.full(self.feature_names.shape[0],
                                           fill_value='e', dtype='str')
                self.relevance_knowledge_str[component.name] = relevance_vector

            # Fill previously created relevance vectors with the relevance scores in defined in the ontology
            for rs in RelevanceScore.instances():
                comp_name = rs.of_component[0].name
                symptom = rs.of_feature[0]
                symptom_index = symptom.has_index_in_data[0]
                relevance_score = rs.has_score[0]

                rel_vec_comp = self.relevance_knowledge_str.get(comp_name)
                rel_vec_comp[symptom_index] = relevance_score

    def update_scores(self, relevance_mapping: dict):
        self.relevance_knowledge_scores = {}

        for comp, str_relevance in self.relevance_knowledge_str.items():
            self.relevance_knowledge_scores[comp] = np.array(
                list(map(lambda x: relevance_mapping.get(x), str_relevance)))


# The ontology is created using the information stored in the dataset and configuration and program code
# for the purpose of simplicity and transparency but could be modeled in any other way
if __name__ == '__main__':
    c = Configuration()

    dataset = Dataset(c.training_data_folder, c)
    dataset.load()
    sm = SemanticModel(c, dataset)
    sm.create(True)
    sm.import_from_file()
