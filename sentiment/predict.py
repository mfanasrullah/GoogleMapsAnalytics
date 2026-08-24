# sentiment/predict.py
import pandas as pd
import joblib
import os
import sys

# Memastikan path direktori utama terbaca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED

class SVMSentimentAnalyzer:
    def __init__(self):
        # Memuat model SVM dan Vectorizer yang sudah Anda latih
        self.model_path = os.path.join(DATA_PROCESSED, '..', 'models', 'svm_model.pkl')
        self.vec_path = os.path.join(DATA_PROCESSED, '..', 'models', 'tfidf_vectorizer.pkl')
        
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vec_path)
        else:
            self.model = None
            self.vectorizer = None
            print("⚠️ Peringatan: Model SVM belum dilatih. Jalankan 'python sentiment/train_model.py' terlebih dahulu.")

    def process_dataframe(self, df, text_column='final_text'):
        """Memproses seluruh baris dataframe dan menebak sentimennya"""
        if self.model is None or self.vectorizer is None:
            print("❌ Prediksi dibatalkan karena model tidak ditemukan.")
            return df
        
        if df.empty:
            return df

        # Pastikan tidak ada data teks yang kosong (NaN)
        texts = df[text_column].fillna('')
        
        # 1. Transformasi teks menjadi angka (TF-IDF)
        X_vec = self.vectorizer.transform(texts)
        
        # 2. Lakukan Prediksi SVM
        predictions = self.model.predict(X_vec)
        
        # 3. Simpan hasil tebakan ke kolom 'sentiment'
        df['sentiment'] = predictions
        
        # Membersihkan label jika keluarannya berupa angka
        df['sentiment'] = df['sentiment'].replace({0: 'NEGATIF', 1: 'NETRAL', 2: 'POSITIF', '-1': 'NEGATIF', '0': 'NEGATIF', '1': 'NETRAL', '2': 'POSITIF'})
        
        return df