# preprocessing.py
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
        
        # 1. Setup Stemmer Sastrawi
        factory_stem = StemmerFactory()
        self.stemmer = factory_stem.create_stemmer()
        
        # 2. Setup Stopword Sastrawi 
        factory_stop = StopWordRemoverFactory()
        self.base_stopwords = set(factory_stop.get_stop_words())
        
        # 3. KATA KUNCI DOMAIN 
        self.custom_stopwords = {
            #  1. Kata Bawaan Sebelumnya 
            'pelabuhan', 'ferry', 'terminal', 'batam', 'centre', 'center', 'sekupang', 
            'punggur', 'telaga', 'nongsapura', 'harbour', 'bay', 'singapore', 'singapura',
            'bintan', 'tanjung', 'pinang', 'kapal', 'tiket', 'nya', 'yg', 'di', 'ke', 'dari', 
            'ini', 'itu', 'untuk', 'dan', 'dengan', 'ada', 'tidak', 'bisa', 'sudah', 'sangat',
            'tempat', 'kalau', 'buat', 'juga', 'aja', 'sih', 'ya', 'yang', 'time', 'dont', 
            'told', 'staff', 'good', 'nice', 'very', 'just', 'only', 'penumpang', 'orang', 
            'menit', 'jam', 'hari',

            #  2. Hasil Stemming Sastrawi (Sering muncul di WordCloud) 
            'labuh', 'tumpang', 'feri', 

            #  3. Pronoun (Kata Ganti Orang) & Tunjuk 
            'anda', 'saya', 'aku', 'dia', 'mereka', 'kami', 'kita', 'kalian', 
            'sini', 'sana', 'situ', 'mana', 'apa', 'siapa', 'kapan',

            #  4. Kata Kerja Umum & Aktivitas Dasar 
            'masuk', 'keluar', 'datang', 'pergi', 'jalan', 'naik', 'turun', 'beli', 
            'bayar', 'minta', 'ambil', 'bawa', 'beri', 'tugas', 'kerja', 'tunggu', 
            'lalu', 'selalu', 'pernah', 'bikin', 'kasih', 'suruh', 'tanya', 'lihat',

            #  5. Adverb (Kata Keterangan) & Penyangkal 
            'sekali', 'paling', 'banyak', 'banget', 'lebih', 'harus', 'cuma', 'hanya', 
            'biar', 'kalo', 'karena', 'karna', 'bukan', 'belum', 'jangan', 'cukup', 'benar',

            #  6. Kata Benda Umum Transportasi & Lokasi 
            'loket', 'konter', 'counter', 'gate', 'gerbang', 'paspor', 'passport', 
            'visa', 'roro', 'uban', 'tanjungpinang', 'pulau', 'malaysia', 'indonesia', 
            'uang', 'barang', 'bagasi', 'dalam', 'atas', 'bawah', 'luar', 'depan', 'belakang',

            #  7. Angka & Waktu 
            'satu', 'dua', 'tiga', 'kali', 'detik', 'bulan', 'tahun', 'waktu', 
            'sekarang', 'nanti', 'besok', 'awal', 'akhir',
            
            # 8. STOPWORDS INGGRIS 
            'the', 'and', 'to', 'is', 'it', 'in', 'for', 'of', 'are', 'you', 'from', 
            'not', 'there', 'this', 'with', 'on', 'at', 'but', 'be', 'as', 'so', 
            'have', 'that', 'or', 'if', 'we', 'was', 'my', 'can', 'will', 'like', 
            'no', 'your', 'about', 'they', 'what', 'which', 'who', 'when', 'where', 
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'nor', 'only', 'own', 'same', 'than', 'too', 'can', 
            'will', 'don', 'should', 'now', 'am', 'were', 'been', 'being', 'has', 'had', 
            'do', 'does', 'did', 'doing', 'an', 'because', 'until', 'while', 'by', 
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here',
            
            # 9. KATA HUBUNG/KETERANGAN INDONESIA TAMBAHAN
            
            'masih', 'mau', 'atau', 'jadi', 'akan', 'saat', 'lagi', 'perlu', 'jika', 
            'lain', 'setelah', 'beberapa', 'sampai', 'oleh', 'tetapi', 'dapat', 'gak', 
            'ga', 'agak', 'adalah', 'semua', 'sekitar', 'seperti', 'salah', 'saja', 
            'pun', 'kok', 'kan', 'dong', 'deh', 'terus', 'padahal', 'bahwa', 'walaupun', 
            'meskipun', 'sedang', 'telah', 'sering', 'kadang', 'jarang', 'pasti', 
            'mungkin', 'boleh', 'wajib', 'serta',
            
            # 10. KATA UMUM DOMAIN PELABUHAN YANG TIDAK MEMILIKI SENTIMEN
            'port', 'area', 'place', 'taxi', 'taksi', 'kota', 'internasional', 
            'international', 'immigration', 'small', 'penyeberangan', 'tujuan', 
            'lantai', 'mobil', 'mall', 'laut', 'ticket', 'ruang', 'lokasi', 'tempatnya', 
            'domestik', 'domestic', 'city', 'trip', 'travel', 'penumpang'
        }
        
        self.all_stopwords = self.base_stopwords.union(self.custom_stopwords)
        
        # 4. Kamus Slang 
        self.slang_dict = {
            'bgus': 'bagus', 'bgt': 'banget', 'bgs': 'bagus', 'brg': 'barang',
            'klo': 'kalau', 'klw': 'kalau', 'gmn': 'bagaimana', 'gmna': 'bagaimana',
            'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
            'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena', 'tp': 'tapi',
            'pdhl': 'padahal', 'jg': 'juga', 'cpt': 'cepat', 'lmbat': 'lambat',
            'lm': 'lama', 'rmh': 'ramah', 'kcwa': 'kecewa', 'jlk': 'jelek', 'mntp': 'mantap',
            'skrg': 'sekarang', 'sy': 'saya', 'ak': 'aku', 'blm': 'belum',
            # Tambahan slang dari review negatif
            'pdhl': 'padahal', 'gajelas': 'tidak jelas', 'trs': 'terus'
        }

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')

    def clean_html_url(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        return text

    def normalize_slang(self, text):
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.all_stopwords]
        return ' '.join(filtered_words)

    def translate_to_indo(self, text):
        try:
            if not text.strip(): return text
            
            lang = detect(text)
            
            if lang != 'id':
                translator = GoogleTranslator(source='auto', target='id')
                return translator.translate(text)
            return text
        except Exception:
            return text

    def clean_text_pipeline(self, text):
        # 1. Lowercase 
        text = str(text).lower()
        
        # 2. Emoji
        text = self.remove_emoji(text)
        
        # 3. URL dan HTML
        text = self.clean_html_url(text)
        
        # 4. Tanda Baca & Angka
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 5. spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Terjemahkan ke Bahasa Indonesia 
        text = self.translate_to_indo(text)
        
        # 7. Normalisasi kata gaul/singkatan 
        text = self.normalize_slang(text.lower()) # di-lower lagi paska terjemah
        
        # 8. kata-kata tidak penting 
        text = self.remove_stopwords(text)
        
        # 9. Stemming 
        text = self.stemmer.stem(text)
        
        return text

    def process_dataframe(self, df, text_col='text'):
        """
        """
        if df.empty:
            return df
            
        print(f"Total data awal: {len(df)} baris")

        if 'location' in df.columns:
            df['location'] = df['location'].replace({
                'Batam Centre Ferry Terminal (A)': 'Batam Centre Ferry Terminal',
                'Batam Centre Ferry Terminal (B)': 'Batam Centre Ferry Terminal'
            })
            print("Penggabungan lokasi Batam Centre selesai dilakukan.")
        
        df = df.drop_duplicates(subset=[text_col])
        print(f"Total data setelah hapus duplikat: {len(df)} baris")
        
        df[text_col] = df[text_col].fillna('')
        
        print(f"Memulai proses pembersihan, terjemahan, dan NLP pada kolom '{text_col}'...")
        print("(Proses ini memakan waktu beberapa menit tergantung jumlah ulasan, karena ada translasi AI dan Stemming)")
        
        df['final_text'] = df[text_col].apply(self.clean_text_pipeline)
        
        df = df[df['final_text'].str.strip() != '']
        print(f"Proses NLP selesai. Total data akhir: {len(df)} baris")
        
        return df

    def process_pipeline(self, df, text_col='text'):
        return self.process_dataframe(df, text_col)
