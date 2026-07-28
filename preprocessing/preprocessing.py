import pandas as pd
import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Agar deteksi bahasa konsisten
DetectorFactory.seed = 0 

class TextPreprocessor:
    def __init__(self):
        print("Mempersiapkan modul NLP Sastrawi dan Kamus...")
        
        # 1. Setup Stemmer Sastrawi (Mengubah kata berimbuhan ke kata dasar)
        factory_stem = StemmerFactory()
        self.stemmer = factory_stem.create_stemmer()
        
        # 2. Setup Stopword Sastrawi (Kata hubung standar Indonesia)
        factory_stop = StopWordRemoverFactory()
        self.base_stopwords = set(factory_stop.get_stop_words())
        
        # ======================================================================
        # 3. KATA KUNCI DOMAIN (CUSTOM STOPWORDS) - SANGAT PENTING UNTUK WORDCLOUD
        # Tambahkan kata-kata yang sering muncul tapi tidak punya makna sentimen di sini
        # ======================================================================
        self.custom_stopwords = {
            'pelabuhan', 'ferry', 'terminal', 'batam', 'centre', 'center', 'sekupang', 
            'punggur', 'telaga', 'nongsapura', 'harbour', 'bay', 'singapore', 'singapura',
            'bintan', 'tanjung', 'pinang', 'kapal', 'tiket', 'nya', 'yg', 'di', 'ke', 'dari', 
            'ini', 'itu', 'untuk', 'dan', 'dengan', 'ada', 'tidak', 'bisa', 'sudah', 'sangat',
            'tempat', 'kalau', 'buat', 'juga', 'aja', 'sih', 'ya', 'yang'
        }
        self.all_stopwords = self.base_stopwords.union(self.custom_stopwords)
        
        # 4. Kamus Slang (Normalisasi bahasa gaul/singkatan)
        self.slang_dict = {
            'bgus': 'bagus', 'bgt': 'banget', 'bgs': 'bagus', 'brg': 'barang',
            'klo': 'kalau', 'klw': 'kalau', 'gmn': 'bagaimana', 'gmna': 'bagaimana',
            'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
            'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena', 'tp': 'tapi',
            'pdhl': 'padahal', 'jg': 'juga', 'cpt': 'cepat', 'lmbat': 'lambat',
            'lm': 'lama', 'rmh': 'ramah', 'kcwa': 'kecewa', 'jlk': 'jelek', 'mntp': 'mantap',
            'skrg': 'sekarang', 'sy': 'saya', 'ak': 'aku', 'blm': 'belum'
        }

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')

    def clean_html_url(self, text):
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Hapus tag HTML
        text = re.sub(r'<.*?>', '', text)
        return text

    def normalize_slang(self, text):
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def remove_stopwords(self, text):
        words = text.split()
        # Buang kata jika ada di dalam daftar all_stopwords
        filtered_words = [word for word in words if word not in self.all_stopwords]
        return ' '.join(filtered_words)

    def translate_to_indo(self, text):
        try:
            # Jika teks kosong, kembalikan kosong
            if not text.strip(): return text
            
            # Deteksi bahasa
            lang = detect(text)
            
            # Jika bukan bahasa Indonesia (id), terjemahkan ke 'id'
            if lang != 'id':
                translator = GoogleTranslator(source='auto', target='id')
                return translator.translate(text)
            return text
        except Exception:
            # Jika gagal deteksi atau terjemah (misal karena jaringan), kembalikan teks asli
            return text

    def clean_text_pipeline(self, text):
        # 1. Lowercase (Kecilkan semua huruf)
        text = str(text).lower()
        
        # 2. Hapus Emoji
        text = self.remove_emoji(text)
        
        # 3. Hapus URL dan HTML
        text = self.clean_html_url(text)
        
        # 4. Hapus Tanda Baca & Angka (Hanya sisakan huruf a-z)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 5. Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Terjemahkan ke Bahasa Indonesia (jika teks asing)
        text = self.translate_to_indo(text)
        
        # 7. Normalisasi kata gaul/singkatan (Slang)
        text = self.normalize_slang(text.lower()) # di-lower lagi paska terjemah
        
        # 8. Hapus kata-kata tidak penting (Stopwords)
        text = self.remove_stopwords(text)
        
        # 9. Stemming (Mengembalikan ke kata dasar, misal: "pelayanan" -> "layan")
        text = self.stemmer.stem(text)
        
        return text

    def process_dataframe(self, df, text_col='text'):
        """
        Fungsi utama yang akan dipanggil di main.py
        """
        if df.empty:
            return df
            
        print(f"Total data awal: {len(df)} baris")
        
        # Menghapus Duplikat berdasarkan teks ulasan
        df = df.drop_duplicates(subset=[text_col])
        print(f"Total data setelah hapus duplikat: {len(df)} baris")
        
        # Mencegah nilai kosong/NaN
        df[text_col] = df[text_col].fillna('')
        
        print(f"Memulai proses pembersihan, terjemahan, dan NLP pada kolom '{text_col}'...")
        print("(Proses ini memakan waktu beberapa menit tergantung jumlah ulasan, karena ada translasi AI dan Stemming)")
        
        # Mengaplikasikan pipeline ke seluruh baris
        # Kita simpan di kolom baru 'final_text' agar teks asli tetap tersimpan (berguna untuk ditampilkan di UI)
        df['final_text'] = df[text_col].apply(self.clean_text_pipeline)
        
        # Hapus baris yang setelah dibersihkan teksnya menjadi kosong sama sekali
        df = df[df['final_text'].str.strip() != '']
        print(f"Proses NLP selesai. Total data akhir: {len(df)} baris")
        
        return df

    # Alias agar sesuai dengan panggilan di main.py Anda
    def process_pipeline(self, df, text_col='text'):
        return self.process_dataframe(df, text_col)