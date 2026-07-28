# sentiment/indobert.py
from transformers import pipeline
import pandas as pd
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="mdhugol/indonesia-bert-sentiment-classification"):
        device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)

    def predict(self, texts):
        # Truncate text to 512 tokens max
        truncated = [str(t)[:512] for t in texts]
        results = self.nlp(truncated)
        return [res['label'] for res in results]

    def process_dataframe(self, df, text_column='final_text'):
        print("Running IndoBERT Sentiment Analysis...")
        df['sentiment'] = self.predict(df[text_column].tolist())
        return df