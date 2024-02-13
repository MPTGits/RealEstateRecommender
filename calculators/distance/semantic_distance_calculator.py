from typing import List

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from facades.text_preprocessing_facade import DataPreprocessorFacade
from calculators.distance.base_distance_calculator import BaseDistanceCalculator


class SemanticDistanceCalculator(BaseDistanceCalculator):

    @classmethod
    def calculate_distance(self, input_text: List[str], compare_texts: List[List[str]], data_preprocessor_facade: DataPreprocessorFacade) -> np.ndarray:
        """
        Calculate the semantic distances between a given input text and a list of texts to compare against.

        This function first vectorizes the input text and the texts to compare using a provided text preprocessor facade.
        It then calculates the cosine similarity between the vectorized input text and each of the vectorized comparison texts,
        effectively measuring the semantic distances between the input text and each text in the comparison list.
        """
        vectorized_input_text = data_preprocessor_facade.vectorize_text(input_text)
        vectorized_compare_texts = [data_preprocessor_facade.vectorize_text(text) for text in compare_texts]
        description_vectors = np.vstack(vectorized_compare_texts)
        input_vector_reshaped = vectorized_input_text.reshape(1, -1)
        semantic_distances = cosine_similarity(input_vector_reshaped, description_vectors).flatten()
        return semantic_distances