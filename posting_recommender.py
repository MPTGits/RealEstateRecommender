import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph

from facades.text_preprocessing_facade import DataPreprocessorFacade
from calculators.distance.area_distance_calculator import AreaDistanceCalculator
from calculators.distance_calculator import DistanceCalculator
from calculators.distance.geo_distance_calculator import GeoDistanceCalculator
from calculators.distance.price_distance_calculator import PriceDistanceCalculator
from calculators.distance.semantic_distance_calculator import SemanticDistanceCalculator

pdfmetrics.registerFont(TTFont('FreeSans', 'fonts/FreeSans.ttf'))


class PostingRecommender:

    def __init__(self, df):
        self.df = df

    def wrap_text(self, text, max_width):
        """
        Splits the text into a list of lines, each not exceeding max_width characters.
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + len(current_line) > max_width:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word)

        lines.append(' '.join(current_line))  # Add the last line
        return lines

    def generate_pdf_report(self, combined_scores, semantic_similarities, numeric_distance_price, numeric_distance_area,
                            geographical_distances_normalized, num_of_postings=5, filename="recommendation_report.pdf"):

        styles = getSampleStyleSheet()
        description_style = ParagraphStyle(
            'DescriptionStyle',
            fontName='FreeSans',
            fontSize=10,
            leading=12  # Adjust line spacing if necessary
        )
        top_indices = combined_scores.argsort()[-num_of_postings:][::-1]

        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        for index in top_indices:
            wrapped_description = self.wrap_text(self.df.iloc[index]['Oписание'], 100)
            formatted_description = "\n".join(wrapped_description)
            description_paragraph = Paragraph(formatted_description, description_style)

            data = [
                ["Метрика", "Стойност"],
                ["Комбинирана сходност", f"{combined_scores[index] * 100:.2f}%"],
                ["Семантична сходност на описанието", f"{semantic_similarities[index] * 100:.2f}%"],
                ["Ценова сходност", f"{numeric_distance_price[index] * 100:.2f}%"],
                ["Размерна сходност", f"{numeric_distance_area[index] * 100:.2f}%"],
                ["Географска сходност", f"{geographical_distances_normalized[index] * 100:.2f}%"],
                ["Oписание", description_paragraph],
                ["Цена", f"{self.df.iloc[index]['Цена']}"],
                ["Размер", f"{self.df.iloc[index]['Размер']}"],
                ["Адрес", self.df.iloc[index]['Адрес']],
            ]

            print(self.df.iloc[index]["Преработено_Описание"])
            print(self.df.iloc[index]["Oписание"])
            print('------------------')

            # Create a table for this posting
            table = Table(data, colWidths=[200, 250])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'FreeSans'),
                ('BOX', (0, 0), (-1, -1), 2, colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            elements.append(table)
            elements.append(Table([[f"{'-' * 100}"]]))  # Separator

        # Build the PDF
        doc.build(elements)


def save_w2vec_model(all_descriptions):
    word2vec_model = Word2Vec(sentences=all_descriptions, vector_size=200, window=5, min_count=50, workers=4)
    model_path = "config/word2vec_real_estate_model.gensim"
    word2vec_model.save(model_path)


if __name__ == "__main__":
    with open('config/sofia_neighbors_map.json', 'r', encoding='utf-8') as file:
        NEIGHBORS_MAP = json.load(file)
    user_area = 60
    user_price = 98000
    user_address = "жк.разсадника,софия"
    user_desc = """Продавам двустаен апартамент на 3 етаж на шпакловка и замазка   с акт  14  до края на годината акт 16"""
    print("Извличане на сходни обяви...")
    df = pd.read_csv('data/all_real_estates.csv')

    df = df.dropna(subset=['Oписание', 'Цена', 'Размер', 'Адрес'])

    data_preprocess_facade = DataPreprocessorFacade(stemmer_file_path="config/stem-context-3.txt",
                                                    model_file_path="config/word2vec_real_estate_model.gensim",
                                                    stop_words_file_path="config/stop_words.txt")

    df = data_preprocess_facade.preprocess_real_estate_data_frame(df)

    save_w2vec_model(df['Преработено_Описание'])

    numeric_distance_area = np.zeros(len(df))
    numeric_distance_price = np.zeros(len(df))
    geographical_distances = np.zeros(len(df))

    dist_calculator = DistanceCalculator()

    dist_calculator.set_calculator(AreaDistanceCalculator())

    for i in range(len(df)):
        numeric_distance_area[i] = dist_calculator.calculate(user_area, df.iloc[i]['Размер'])

    dist_calculator.set_calculator(PriceDistanceCalculator())

    for i in range(len(df)):
        numeric_distance_price[i] = dist_calculator.calculate(user_price, df.iloc[i]['Цена'])

    dist_calculator.set_calculator(GeoDistanceCalculator())

    for i in range(len(df)):
        geographical_distances[i] = dist_calculator.calculate(user_address, df.iloc[i]['Преработен_Адрес'],
                                                              NEIGHBORS_MAP)

    geographical_distances_normalized = 1 - (geographical_distances / np.max(geographical_distances))

    dist_calculator.set_calculator(SemanticDistanceCalculator())

    preprocessed_user_desc = data_preprocess_facade.preprocess_text(user_desc)

    print(preprocessed_user_desc)

    semantic_similarities = dist_calculator.calculate(preprocessed_user_desc, df['Преработено_Описание'], data_preprocess_facade)


    combined_scores = 0.25 * semantic_similarities + 0.25 * numeric_distance_area + 0.25 * numeric_distance_price + 0.25 * geographical_distances_normalized

    PostingRecommender(df).generate_pdf_report(combined_scores,
                                               semantic_similarities,
                                               numeric_distance_price,
                                               numeric_distance_area,
                                               geographical_distances_normalized,
                                               num_of_postings=5,
                                               filename="recommendation_report.pdf")
