
import pandas as pd
import os
import sys


from config import DATA_RAW, DATA_PROCESSED


from scraper.scraper import GoogleMapsScraper
from preprocessing.preprocessing import TextPreprocessor
from sentiment.indobert import SentimentAnalyzer
from aspect.aspect import AspectExtractor


try:
    from sentiment.evaluation import SentimentEvaluator
except ImportError:
    SentimentEvaluator = None

def run_pipeline():
    print("="*50)
    print("🛳️  PIPELINE DATA ANALYTICS PELABUHAN BATAM 🛳️")
    print("="*50)
    print("Pilih mode eksekusi:")
    print("1. Mulai dari Awal (Scraping Data Baru -> Preprocessing -> Analisis)")
    print("2. Gunakan Data Mentah (Skip Scraping -> Preprocessing -> Analisis)")
    print("="*50)
    
    pilihan = input("Masukkan pilihan Anda (1 atau 2): ").strip()
    

    df = pd.DataFrame()


    if pilihan == '1':
        print("\n[1/5] Menjalankan Scraper Google Maps...")
        scraper = GoogleMapsScraper()
        df = scraper.run_all()
        
        if df is None or df.empty:
            print("❌ Scraping gagal atau tidak ada data yang diambil. Program dihentikan.")
            return
            
    elif pilihan == '2':
        print("\n[1/5] Membaca data mentah (raw_reviews.csv) yang sudah ada...")
        raw_path = os.path.join(DATA_RAW, 'raw_reviews.csv')
        
        if not os.path.exists(raw_path):
            print(f"❌ File mentah tidak ditemukan di: {raw_path}")
            print("Silakan jalankan ulang program dan pilih Mode 1 (Scraping) terlebih dahulu.")
            return
            
        df = pd.read_csv(raw_path)
        print(f"✅ Berhasil memuat {len(df)} baris data mentah.")
        
    else:
        print("❌ Pilihan tidak valid. Harap masukkan angka 1 atau 2. Program dihentikan.")
        return


    print("\n[2/5] Memulai Preprocessing Teks (NLP)...")
    preprocessor = TextPreprocessor()

    df = preprocessor.process_pipeline(df, text_col='text')


    print("\n[3/5] Memulai Prediksi Sentimen AI (IndoBERT)...")
    try:
        sentiment_analyzer = SentimentAnalyzer()

        df = sentiment_analyzer.process_dataframe(df, text_column='semi_clean_text') 
        print("✅ Analisis Sentimen Selesai.")
    except Exception as e:
        pass


    print("\n[4/5] Memulai Ekstraksi Aspek Keluhan/Pujian...")
    try:
        aspect_extractor = AspectExtractor(method='rule-based')

        df = aspect_extractor.process_dataframe(df, text_column='final_text')
        print("✅ Ekstraksi Aspek Selesai.")
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal memproses aspek. Error: {e}")


    os.makedirs(DATA_PROCESSED, exist_ok=True)
    final_path = os.path.join(DATA_PROCESSED, 'final_dataset.csv')
    df.to_csv(final_path, index=False)
    print(f"\n🎉 PIPELINE SELESAI! Data final berhasil diperbarui dan disimpan di: {final_path}")


    if SentimentEvaluator is not None:
        print("\n[5/5] Mengukur Akurasi dan Evaluasi Model...")
        try:
            evaluator = SentimentEvaluator()
            evaluator.run_evaluation()
        except Exception as e:
            print(f"⚠️ Evaluasi gagal dijalankan. Error: {e}")
    else:
        print("\n[5/5] Evaluasi dilewati (File sentiment/evaluation.py tidak ditemukan).")

if __name__ == "__main__":
    run_pipeline()
