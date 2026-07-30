import pandas as pd
import os
from config import DATA_PROCESSED

from scraper.scraper import GoogleMapsScraper
from preprocessing.preprocessing import TextPreprocessor
from sentiment.indobert import SentimentAnalyzer

def run_pipeline():
    print("=== MULAI PIPELINE DATA ANALYTICS ===")
    
    print("\n[1/3] Menjalankan Scraper Google Maps...")
    scraper = GoogleMapsScraper()
    df_raw = scraper.run_all() 
    
    if df_raw.empty:
        print("Data kosong! Pastikan koneksi internet stabil dan elemen Google Maps tidak berubah.")
        return
        
    print("\n[2/3] Membersihkan dan Menerjemahkan Teks...")
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.process_pipeline(df_raw, text_col='text')
    
    print("\n[3/3] Menganalisis Sentimen dengan AI...")
    analyzer = SentimentAnalyzer()
    df_final = analyzer.process_dataframe(df_clean, text_column='final_text')
    
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED, "final_dataset.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"\n=== PIPELINE SELESAI! Data tersimpan di {output_path} ===")
    print("Sekarang Anda bisa menjalankan: streamlit run app.py")

if __name__ == "__main__":
    run_pipeline()
