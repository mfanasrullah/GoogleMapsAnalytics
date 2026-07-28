# topic/bertopic.py
from bertopic import BERTopic
import pandas as pd

class TopicModeler:
    def __init__(self):
        # Menggunakan model multilingual ringan yang mendukung Bahasa Indonesia
        self.topic_model = BERTopic(language="multilingual", calculate_probabilities=True)

    def fit_transform(self, docs):
        print("Training BERTopic...")
        topics, probs = self.topic_model.fit_transform(docs)
        return topics, self.topic_model

    def get_topic_info(self):
        return self.topic_model.get_topic_info()