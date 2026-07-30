import pandas as pd
from .keyword import ASPECT_DICT

class AspectExtractor:
    def __init__(self, method='rule-based'):
        """
        method: 'rule-based' atau 'zero-shot'
        """
        self.method = method
        
        if self.method == 'zero-shot':
            print("Loading Multilingual Zero-Shot Model...")
            from transformers import pipeline
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="joeddav/xlm-roberta-large-xnli"
            )
            self.candidate_labels = list(ASPECT_DICT.keys())

    def extract_rule_based(self, text):
        text = str(text).lower()
        detected_aspects = []
        
        for aspect, keywords in ASPECT_DICT.items():
            if any(keyword in text for keyword in keywords):
                detected_aspects.append(aspect)
                
        return detected_aspects if detected_aspects else ["Umum/Lainnya"]

    def extract_zero_shot(self, text):
        text = str(text)[:512] # Batasi panjang teks untuk model transformer
        
        result = self.classifier(text, self.candidate_labels)
        
        detected_aspects = [
            label for label, score in zip(result['labels'], result['scores']) 
            if score > 0.4
        ]
        
        return detected_aspects if detected_aspects else ["Umum/Lainnya"]

    def process_dataframe(self, df, text_column='clean_text'):
        print(f"Mengekstraksi aspek menggunakan metode: {self.method}...")
        
        if self.method == 'rule-based':
            df['aspects'] = df[text_column].apply(self.extract_rule_based)
        elif self.method == 'zero-shot':
            df['aspects'] = df[text_column].apply(self.extract_zero_shot)
            
        return df
