import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TARGET_LOCATIONS, DATA_RAW

class GoogleMapsScraper:
    def __init__(self):
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless=false')
        # SANGAT PENTING: Memaksa ukuran layar jadi Full HD agar tab Ulasan tidak tersembunyi
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=id')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')

        # ==============================================================
        # PARAMETER ANTI-CRASH & STABILISASI (Tanpa Profil Pribadi)
        # ==============================================================
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu') # Meringankan beban kartu grafis
        
        # OPSI DI BAWAH INI DIMATIKAN AGAR TIDAK CRASH / NOT REACHABLE
        # options.add_argument('--remote-debugging-port=9222')
        # user_data_path = r"C:\Users\user\AppData\Local\Google\Chrome\User Data"
        # options.add_argument(f"--user-data-dir={user_data_path}")
        # options.add_argument("--profile-directory=Profile 1")
        # ==============================================================
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def scrape_location(self, location_name, url, max_scrolls=10):
        if location_name == "Pemanasan Browser":
            return []
            
        self.driver.get(url)
        wait = WebDriverWait(self.driver, 30)

        # ==============================================================
        # [BARU] INTERVENSI MANUAL KHUSUS LOKASI YANG SENSITIF (3 MENIT)
        # ==============================================================
        if "Batam Centre" in location_name or "Batam Center" in location_name:
            print(f"\n⚠️  PERHATIAN: Memasuki halaman {location_name}.")
            print("Sistem Google mungkin meminta Login atau Persetujuan Cookie.")
            print("Anda memiliki waktu 3 menit (100 detik) untuk menyelesaikannya secara manual di browser...")
            
            # Hitung mundur 100 detik (3 menit)
            for i in range(100, 0, -1):
                mins, secs = divmod(i, 60)
                print(f"Sisa waktu intervensi: {mins:02d} menit {secs:02d} detik", end='\r')
                time.sleep(1)
            print("\n✔️  Waktu intervensi selesai. Mengambil alih kontrol browser kembali...\n")
        else:
            # Berikan waktu standar 5 detik untuk lokasi lain agar halaman ter-render sempurna
            time.sleep(5)
        # ==============================================================

        # 1. Tunggu dan klik tab "Ulasan"
        try:
            xpath_ulasan = """
            //button[
                contains(@aria-label, 'Ulasan') or
                contains(@aria-label, 'Reviews') or
                .//div[contains(text(), 'Ulasan')] or
                .//div[contains(text(), 'Reviews')]
            ]
            """
            reviews_btn = wait.until(EC.presence_of_element_located((By.XPATH, xpath_ulasan)))
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", reviews_btn)
            time.sleep(1)
            self.driver.execute_script("arguments[0].click();", reviews_btn)
            time.sleep(4)
        except Exception as e:
            print(f"[ERROR] Tab ulasan '{location_name}' tetap tidak ditemukan atau terblokir Captcha.")
            return []

        # 2. Logic Scroll (Diperlambat sedikit menjadi 3 detik agar lebih stabil saat ambil data banyak)
        print(f"Menggulir ulasan {location_name} (Max: {max_scrolls} kali)...")
        for i in range(max_scrolls):
            try:
                reviews_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")
                if reviews_elements:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", reviews_elements[-1])
                    time.sleep(3)
            except:
                break
                
        # 3. Klik semua tombol "Lainnya" / "Selengkapnya"
        try:
            more_buttons = self.driver.find_elements(By.XPATH, "//button[contains(@aria-label, 'selengkapnya') or contains(text(), 'Lainnya') or contains(text(), 'More')]")
            for btn in more_buttons:
                self.driver.execute_script("arguments[0].click();", btn)
            time.sleep(4)
        except:
            pass

        # ==============================================================
        # 4. PARSE DATA MENGGUNAKAN BEAUTIFULSOUP (UPGRADED)
        # ==============================================================
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='jftiEf')
        
        data = []
        for r in reviews:
            try:
                # 4a. Teks Ulasan (Yang sudah ada)
                text_elem = r.select_one('.wiI7pd, .MyEned')
                text = text_elem.text if text_elem else ""
                
                # 4b. Rating & Waktu (Yang sudah ada)
                rating_elem = r.select_one('.kvMYJc')
                rating = rating_elem.get('aria-label') if rating_elem else ""
                time_elem = r.select_one('.rsqaWe')
                time_posted = time_elem.text if time_elem else ""
                
                # ---------------- FITUR BARU ----------------
                
                # 4c. Nama Reviewer
                name_elem = r.select_one('.d4r55')
                reviewer_name = name_elem.text.strip() if name_elem else "Anonim"
                
                # 4d. Status Local Guide & Jumlah Kontribusi
                contrib_elem = r.select_one('.RfnDt')
                contrib_text = contrib_elem.text.strip() if contrib_elem else ""
                # Deteksi boolean True/False untuk Local Guide
                is_local_guide = True if "Local Guide" in contrib_text else False
                
                # 4e. Jumlah Like (Suka)
                # Mencari angka di dalam tombol jempol
                like_elem = r.select_one('.kX08se, .pkWtMe')
                likes = like_elem.text.strip() if like_elem and like_elem.text.strip().isdigit() else "0"
                
                # 4f. Respon dari Pemilik (Owner Response)
                owner_elem = r.select_one('.CDe7Nb')
                owner_response = owner_elem.text.strip() if owner_elem else ""
                
                # 4g. Apakah melampirkan Foto? (Boolean True/False)
                # Class .Tya61d adalah kontainer foto thumbnail di ulasan Maps
                photo_elems = r.select('.Tya61d')
                has_photo = True if len(photo_elems) > 0 else False
                
                # --------------------------------------------

                # Simpan data hanya jika ada teks ulasannya
                if text.strip():
                    data.append({
                        'location': location_name,
                        'reviewer_name': reviewer_name,
                        'is_local_guide': is_local_guide,
                        'contributor_info': contrib_text,
                        'rating': rating,
                        'time': time_posted,
                        'text': text,
                        'likes': int(likes), # Konversi string "12" jadi angka 12
                        'has_photo': has_photo,
                        'owner_response': owner_response
                    })
            except Exception as e:
                # Jika 1 ulasan gagal, lewati ke ulasan berikutnya agar script tidak mati
                continue
                
        # Menghapus data duplikat (mengubah list of dict jadi kumpulan tuple unik, lalu dikembalikan ke dict)
        data = [dict(t) for t in {tuple(d.items()) for d in data}]
        
        print(f"✅ Berhasil mengambil {len(data)} ulasan (FULL METADATA) dari {location_name}.")
        return data

    def run_all(self):
        all_data = []
        os.makedirs(DATA_RAW, exist_ok=True)
        output_path = os.path.join(DATA_RAW, 'raw_reviews.csv')
        
        for loc_name, url in TARGET_LOCATIONS.items():
            print(f"\nMulai proses: {loc_name}...")
            # Setel max_scrolls ke angka besar jika ingin data banyak
            data = self.scrape_location(loc_name, url, max_scrolls=100)
            all_data.extend(data)
            
        if not all_data:
            print("❌ Gagal mengambil ulasan dari semua pelabuhan.")
            self.driver.quit()
            return pd.DataFrame()

        df_new = pd.DataFrame(all_data)
        
        # ==========================================================
        # LOGIKA PENGGABUNGAN DATA (APPEND & HAPUS DUPLIKAT)
        # ==========================================================
        if os.path.exists(output_path):
            print(f"\n🔄 Menemukan data historis. Menggabungkan data baru dengan data lama...")
            df_old = pd.read_csv(output_path)
            
            # Gabungkan data lama dan data baru
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # Hapus ulasan yang sama persis (Berdasarkan Nama, Teks, dan Lokasi)
            # keep='last' memastikan kita menyimpan metadata scraping yang paling baru
            df_combined = df_combined.drop_duplicates(subset=['reviewer_name', 'text', 'location'], keep='last')
            
            print(f"📈 Total data sebelumnya: {len(df_old)} baris")
            print(f"📈 Total data setelah digabung: {len(df_combined)} baris")
            df_final = df_combined
        else:
            df_final = df_new
            print(f"📈 Total data baru: {len(df_final)} baris")

        # Simpan kembali menimpa file lama dengan versi yang sudah diperkaya
        df_final.to_csv(output_path, index=False)
        self.driver.quit()
        return df_final

if __name__ == "__main__":
    scraper = GoogleMapsScraper()
    scraper.run_all()