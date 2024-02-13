import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
import string
from bulstem.stem import BulStemmer


class DataPreprocessorFacade:

    def __init__(self, stemmer_file_path, model_file_path, stop_words_file_path=None):
        self.stemmer = BulStemmer.from_file(stemmer_file_path, min_freq=2, left_context=1)
        self.model = Word2Vec.load(model_file_path)
        if stop_words_file_path:
            self._load_stop_words(stop_words_file_path)
        else:
            self.stop_words = set()

    def _load_stop_words(self, stop_words_file_path):
        with open(stop_words_file_path, "r", encoding="utf-8") as file:
            self.stop_words =  {line.strip() for line in file}

    def tokenize_text(self, text: str):
        return word_tokenize(text)

    def remove_punctuation(self, text: str):
        return "".join(char for char in text if char not in string.punctuation and not char.isnumeric())

    def remove_stop_words(self, text: list):
        return [self.stemmer.stem(word) for word in text if word and self.stemmer.stem(word) not in self.stop_words]

    def remove_english_words(self, text: list):
        """Useful when we are preprocessing a text that is not in English."""
        return [word for word in text if not re.match(r'^[a-zA-Z]+$', word)]

    def vectorize_text(self, preprocessed_text: list):
        vectors = [self.model.wv[word] for word in preprocessed_text if word in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def preprocess_real_estate_data_frame(self, df: pd.DataFrame):
        df['Преработен_Адрес'] = df['Адрес'].str.replace(" ", '').str.lower()
        df['Размер'] = df['Размер'].str.replace('m²', '').astype(float)
        df['Цена'] = df['Цена'].str.replace(',', '').astype(float)
        df['Преработено_Описание'] = df['Oписание'].apply(self.preprocess_text)
        return df

    def preprocess_text(self, text: str):
        lowered_text = text.lower().replace("\n", " ")
        tokenized_text = self.tokenize_text(lowered_text)
        no_punctuation_text = [self.remove_punctuation(word) for word in tokenized_text]
        no_stop_words_text = self.remove_stop_words(no_punctuation_text)
        no_english_words_text = self.remove_english_words(no_stop_words_text)
        return list(set(no_english_words_text))