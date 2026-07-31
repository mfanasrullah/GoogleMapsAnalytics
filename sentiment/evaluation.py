# sentiment/evaluation.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Memastikan path direktori utama terbaca (jika dieksekusi dari dalam folder sentiment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED

class SentimentEvaluator:
    def __init__(self):
        # Membaca data yang sudah diproses
        self.file_path = os.path.join(DATA_PROCESSED, 'final_dataset.csv')
        self.df = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ File tidak ditemukan: {self.file_path}")
            return None
        return pd.read_csv(self.file_path)

    def _create_ground_truth(self):
        """Membuat standar kebenaran (Ground Truth) berdasarkan Rating Bintang"""
        if 'rating' not in self.df.columns:
            print("❌ Kolom 'rating' tidak ditemukan untuk membuat Ground Truth.")
            return False

        # Mengekstrak angka dari teks (misal: "5 bintang" -> 5.0)
        self.df['rating_num'] = self.df['rating'].astype(str).str.extract(r'(\d+)').astype(float)
        
        def map_rating(r):
            if pd.isna(r): return 'NETRAL'
            if r >= 4: return 'POSITIF'
            elif r <= 2: return 'NEGATIF'
            else: return 'NETRAL'
            
        self.df['ground_truth'] = self.df['rating_num'].apply(map_rating)
        return True

    def _clean_predictions(self):
        """Menyamakan format label hasil prediksi AI (IndoBERT)"""
        if 'sentiment' not in self.df.columns:
            print("❌ Kolom 'sentiment' tidak ditemukan.")
            return False
            
        label_map = {
            'LABEL_0': 'POSITIF', 'LABEL_1': 'NETRAL', 'LABEL_2': 'NEGATIF',
            'POSITIVE': 'POSITIF', 'NEUTRAL': 'NETRAL', 'NEGATIVE': 'NEGATIF'
        }
        self.df['model_prediction'] = self.df['sentiment'].map(lambda x: label_map.get(str(x).upper(), 'NETRAL'))
        return True

    def run_evaluation(self):
        """Menjalankan seluruh proses evaluasi"""
        if self.df is None or not self._create_ground_truth() or not self._clean_predictions():
            return

        print("\n" + "="*50)
        print("📊 HASIL EVALUASI SENTIMEN MODEL AI")
        print("="*50)

        # 1. Classification Report (Akurasi, Presisi, Recall)
        report = classification_report(self.df['ground_truth'], self.df['model_prediction'])
        print("\n1. CLASSIFICATION REPORT:")
        print(report)

        # 2. Confusion Matrix
        print("\n2. MENYIMPAN CONFUSION MATRIX...")
        cm = confusion_matrix(self.df['ground_truth'], self.df['model_prediction'], labels=['NEGATIF', 'NETRAL', 'POSITIF'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['NEGATIF', 'NETRAL', 'POSITIF'], 
                    yticklabels=['NEGATIF', 'NETRAL', 'POSITIF'])
        plt.title('Confusion Matrix: Prediksi AI vs Rating Pengguna')
        plt.ylabel('Ground Truth (Asli dari Pengguna)')
        plt.xlabel('Prediksi Model AI')
        
        # Menyimpan grafik ke folder data/processed
        cm_path = os.path.join(DATA_PROCESSED, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Grafik Confusion Matrix disimpan di: {cm_path}")

        # 3. Error Analysis
        print("\n3. ANALISIS KESALAHAN (ERROR ANALYSIS)...")
        errors_df = self.df[self.df['ground_truth'] != self.df['model_prediction']].copy()
        
        # Memilih kolom yang relevan untuk dianalisis
        cols_to_save = ['location', 'rating', 'ground_truth', 'model_prediction', 'text', 'final_text']
        errors_df = errors_df[[c for c in cols_to_save if c in errors_df.columns]]
        
        error_path = os.path.join(DATA_PROCESSED, 'error_analysis.csv')
        errors_df.to_csv(error_path, index=False)
        
        print(f"✅ Terdapat {len(errors_df)} data yang salah prediksi (Missclassified).")
        print(f"✅ Detail kesalahan telah disimpan di: {error_path}")
        
        print("\nContoh Kasus Kesalahan (5 Data Teratas):")
        print(errors_df[['ground_truth', 'model_prediction', 'text']].head(5).to_string())

if __name__ == "__main__":
    evaluator = SentimentEvaluator()
    evaluator.run_evaluation()