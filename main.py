# main.py
import pandas as pd
import os
import sys
import argparse # Modul untuk menangkap argumen dari Terminal/GitHub Actions

from config import DATA_RAW, DATA_PROCESSED
from scraper.scraper import GoogleMapsScraper
from preprocessing.preprocessing import TextPreprocessor
from aspect.aspect import AspectExtractor

# [PERBAIKAN]: Menggunakan model SVM yang baru, bukan IndoBERT lama
from sentiment.predict import SVMSentimentAnalyzer 

try:
    from sentiment.evaluation import SentimentEvaluator
except ImportError:
    SentimentEvaluator = None

def run_pipeline(mode=None):
    print("="*50)
    print("🛳️  PIPELINE DATA ANALYTICS PELABUHAN BATAM 🛳️")
    print("="*50)
    
    # Jika dijalankan manual oleh Anda di laptop
    if mode is None:
        print("Pilih mode eksekusi:")
        print("1. Mulai dari Awal (Scraping -> Preprocessing -> Analisis)")
        print("2. Gunakan Data Mentah (Skip Scraping -> Preprocessing -> Analisis)")
        print("3. Auto-Update (Otomatis skip scraping, cocok untuk GitHub Actions)")
        print("="*50)
        pilihan = input("Masukkan pilihan Anda (1, 2, atau 3): ").strip()
    else:
        # Jika dijalankan oleh GitHub Actions, otomatis pakai mode yang diset
        pilihan = str(mode)

    df = pd.DataFrame()

    # ==========================================
    # FASE 1: PENGUMPULAN DATA
    # ==========================================
    if pilihan == '1':
        print("\n[1/5] Menjalankan Scraper Google Maps...")
        scraper = GoogleMapsScraper()
        df = scraper.run_all()
        
        if df is None or df.empty:
            print("❌ Scraping gagal atau tidak ada data yang diambil. Program dihentikan.")
            return
            
    elif pilihan in ['2', '3']:
        print("\n[1/5] Membaca data mentah gabungan (raw_reviews.csv)...")
        raw_path = os.path.join(DATA_RAW, 'raw_reviews.csv')
        
        if not os.path.exists(raw_path):
            print(f"❌ File mentah tidak ditemukan di: {raw_path}")
            return
            
        df = pd.read_csv(raw_path)
        print(f"✅ Berhasil memuat {len(df)} baris data mentah.")
        
    else:
        print("❌ Pilihan tidak valid. Program dihentikan.")
        return

    # ==========================================
    # FASE 2: PREPROCESSING (PEMBERSIHAN TEKS)
    # ==========================================
    print("\n[2/5] Memulai Preprocessing Teks (NLP)...")
    preprocessor = TextPreprocessor()
    df = preprocessor.process_pipeline(df, text_col='text')

    # ==========================================
    # FASE 3: ANALISIS SENTIMEN (SVM CUSTOM)
    # ==========================================
    print("\n[3/5] Memulai Prediksi Sentimen (SVM + TF-IDF)...")
    try:
        svm_analyzer = SVMSentimentAnalyzer()
        # [PENTING]: Model SVM Anda di train_model.py dilatih menggunakan teks 'final_text'
        df = svm_analyzer.process_dataframe(df, text_column='final_text') 
        print("✅ Analisis Sentimen Selesai.")
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal memproses sentimen. Error: {e}")

    # ==========================================
    # FASE 4: EKSTRAKSI ASPEK KELUHAN/PUJIAN
    # ==========================================
    print("\n[4/5] Memulai Ekstraksi Aspek Keluhan/Pujian...")
    try:
        aspect_extractor = AspectExtractor(method='rule-based')
        df = aspect_extractor.process_dataframe(df, text_column='final_text')
        print("✅ Ekstraksi Aspek Selesai.")
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal memproses aspek. Error: {e}")

    # ==========================================
    # PENYIMPANAN DATA FINAL
    # ==========================================
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    final_path = os.path.join(DATA_PROCESSED, 'final_dataset.csv')
    df.to_csv(final_path, index=False)
    print(f"\n🎉 PIPELINE SELESAI! Data final berhasil diperbarui dan disimpan di: {final_path}")

    # ==========================================
    # FASE 5: EVALUASI AKURASI (Tingkat Lanjut)
    # ==========================================
    # Jika mode 3 (Otomatis di GitHub), lewati print evaluasi agar lebih cepat
    if pilihan != '3':
        if SentimentEvaluator is not None:
            print("\n[5/5] Mengukur Akurasi dan Evaluasi Model...")
            try:
                evaluator = SentimentEvaluator()
                evaluator.run_evaluation()
            except Exception as e:
                print(f"⚠️ Evaluasi gagal dijalankan. Error: {e}")
        else:
            print("\n[5/5] Evaluasi dilewati (File evaluation.py tidak ditemukan).")
    else:
        print("\n[5/5] Evaluasi dilewati (Mode Automasi GitHub Actions).")

if __name__ == "__main__":
    # Setup agar script bisa menerima argumen saat dijalankan dari GitHub
    parser = argparse.ArgumentParser(description="Jalankan Pipeline Analitik Pelabuhan")
    parser.add_argument('--auto', action='store_true', help='Jalankan mode otomatis tanpa input user')
    args = parser.parse_args()
    
    if args.auto:
        # Menjalankan Mode 3 secara otomatis (Skip scraping, baca raw_reviews, tanpa input)
        run_pipeline(mode=3)
    else:
        # Menjalankan secara normal (Meminta ketikan 1 atau 2)
        run_pipeline()