# sentiment/indobert.py
from transformers import pipeline
import pandas as pd
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="mdhugol/indonesia-bert-sentiment-classification"):
        device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)
        
        # PERBAIKAN: Kamus pemetaan label khusus untuk model mdhugol
        self.label_mapping = {
            "LABEL_0": "POSITIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "NEGATIVE"
        }

    def predict(self, texts):
        # Truncate text to 512 tokens max
        truncated = [str(t)[:512] for t in texts]
        results = self.nlp(truncated)
        
        # Ubah output mesin (LABEL_0/1/2) menjadi teks yang benar menggunakan label_mapping
        mapped_labels = []
        for res in results:
            raw_label = res['label']
            mapped_labels.append(self.label_mapping.get(raw_label, raw_label))
            
        return mapped_labels

    def process_dataframe(self, df, text_column='final_text'):
        print("Running IndoBERT Sentiment Analysis...")
        df['sentiment'] = self.predict(df[text_column].tolist())
        return df
