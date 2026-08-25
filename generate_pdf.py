from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Warna teks biru tua (mirip tema ocean-deep)
        self.set_text_color(10, 38, 71)
        # Judul
        self.cell(0, 10, 'Dokumentasi Fitur: Dashboard Analisis Sentimen Pelabuhan', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, 'Powered by Tim Analitik Polibatam', 0, 1, 'C')
        # Garis bawah
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        # Posisi di 1.5 cm dari bawah
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        # Nomor halaman
        self.cell(0, 10, 'Halaman ' + str(self.page_no()) + ' / {nb}', 0, 0, 'C')

    def chapter_title(self, num, title):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Warna latar belakang (mirip tide-teal)
        self.set_fill_color(44, 159, 163)
        self.set_text_color(255, 255, 255)
        # Judul Bab
        self.cell(0, 10, 'Bagian %d: %s' % (num, title), 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        # Arial 11
        self.set_font('Arial', '', 11)
        self.set_text_color(23, 41, 61) # Ink color
        # Teks justify
        self.multi_cell(0, 7, body)
        self.ln()

    def print_chapter(self, num, title, body):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(body)

# --- Inisiasi Pembuatan PDF ---
pdf = PDF()
pdf.alias_nb_pages()

# --- Konten Halaman ---

# Pendahuluan
intro_text = """Dashboard Analisis Sentimen Pelabuhan adalah aplikasi berbasis web interaktif yang dikembangkan menggunakan Streamlit. Aplikasi ini bertujuan untuk memantau, menganalisis, dan memvisualisasikan ulasan pelanggan (penumpang) dari berbagai terminal feri di Batam secara real-time.

Sistem ini didukung oleh kecerdasan buatan (Support Vector Machine / SVM) yang secara otomatis mengklasifikasikan setiap ulasan ke dalam sentimen Positif, Netral, atau Negatif, serta mengekstraksi aspek spesifik yang dibicarakan (misalnya: Pelayanan, Kebersihan, Imigrasi, dll)."""
pdf.print_chapter(1, 'Pendahuluan & Tujuan Sistem', intro_text)

# Fitur Sidebar & Filter
sidebar_text = """Menu navigasi utama (Sidebar) berada di sisi kiri layar. Fitur ini dirancang untuk memberikan kontrol penuh kepada pengguna atas data yang ditampilkan.

1. Tombol Segarkan Data: Memaksa sistem untuk menghapus cache (memori sementara) dan memuat ulang data ulasan serta model AI terbaru dari server.
2. Filter Pelabuhan (Multiselect): Pengguna dapat memilih untuk menampilkan data dari satu pelabuhan spesifik (misal: Batam Centre saja) atau membandingkan beberapa pelabuhan sekaligus.
3. Filter Rentang Tanggal: Terdapat kalender interaktif untuk menyaring ulasan berdasarkan waktu (Tanggal Mulai dan Tanggal Akhir).
4. Tombol Toggle Layar Penuh: Menyembunyikan sidebar agar area visualisasi data menjadi lebih luas dan fokus."""
pdf.print_chapter(2, 'Kontrol Navigasi (Sidebar & Filter)', sidebar_text)

# KPI Utama
kpi_text = """Tepat di bawah banner judul, terdapat tiga metrik utama (Key Performance Indicators) yang langsung memberikan ringkasan status pelabuhan berdasarkan filter yang aktif:

1. Total Volume Ulasan: Menampilkan jumlah keseluruhan ulasan yang masuk ke dalam sistem.
2. Rata-rata Rating (Bintang): Mengkalkulasi nilai rata-rata kepuasan pelanggan (skala 1.0 hingga 5.0).
3. Pelabuhan Teranalisis: Menunjukkan jumlah lokasi pelabuhan yang sedang dianalisis secara bersamaan."""
pdf.print_chapter(3, 'Indikator Kinerja Utama (KPI Card)', kpi_text)

# Tab 1: Visualisasi Data
tab1_text = """Tab ini menyajikan grafik interaktif untuk melihat tren dan perbandingan antar pelabuhan:

1. Diagram Batang (Distribusi Popularitas): Membandingkan pelabuhan mana yang menerima lalu lintas ulasan terbanyak. Setiap pelabuhan direpresentasikan dengan warna unik yang konsisten.
2. Diagram Sebar (Kualitas vs Volume): Memetakan pelabuhan pada sumbu X (Jumlah Ulasan) dan sumbu Y (Rata-rata Rating). Ukuran gelembung (bubble) mewakili besarnya volume. Grafik ini sangat berguna untuk melihat apakah pelabuhan yang sibuk mampu mempertahankan pelayanannya.
3. Grafik Garis (Tren Volume per Bulan): Melacak fluktuasi jumlah ulasan dari waktu ke waktu.
4. Diagram Batang Keluhan Aspek: Memecah sentimen pelanggan berdasarkan 10 aspek spesifik (Fasilitas, Kebersihan, Imigrasi, Harga, dll).
5. Grafik Garis Prediktif: Memantau tren perbincangan suatu aspek tertentu dari bulan ke bulan.

*Catatan: Setiap diagram dilengkapi tombol "Perbesar" untuk melihat grafik dalam jendela pop-up ukuran penuh (Full-Screen)."""
pdf.print_chapter(4, 'Tab 1 - Analitik & Visualisasi Data', tab1_text)

# Tab 2: WordCloud & Heatmap
tab2_text = """Tab ini dirancang untuk menemukan akar masalah secara kualitatif dan spasial.

1. WordCloud (Awan Kata): Menampilkan kata-kata yang paling sering diketik oleh penumpang. Kata yang sering muncul akan berukuran lebih besar. Filter Stopwords Sastrawi telah membuang kata-kata tidak bermakna (seperti 'dan', 'di') serta pesan error sistem ('server', 'error').
2. Heatmap Keluhan Konsumen: Matriks warna (merah muda hingga merah tua) yang melacak titik waktu krisis. Sumbu X menampilkan Periode Bulan, dan Sumbu Y menampilkan Nama Pelabuhan. Warna merah tua mengindikasikan lonjakan keluhan (Rating 1 & 2) yang tinggi pada bulan tersebut."""
pdf.print_chapter(5, 'Tab 2 - Pemrosesan Bahasa Alami (WordCloud & Heatmap)', tab2_text)

# Tab 3: Uji Sentimen Real-Time
tab3_text = """Sistem simulasi langsung (Playground) untuk menguji kecerdasan model AI.

Pengguna dapat mengetikkan teks kalimat bebas (keluhan atau pujian) ke dalam kolom yang disediakan. Sistem akan secara instan membersihkan teks tersebut (Stemming Sastrawi), mengubahnya menjadi angka matematis (TF-IDF Vectorizer), dan meminta model Support Vector Machine (SVM) untuk menebak:
Apakah teks tersebut POSITIF (Hijau), NETRAL (Abu-abu), atau NEGATIF (Merah). 

Sistem juga menampilkan persentase probabilitas (Tingkat Keyakinan AI) untuk setiap tebakannya."""
pdf.print_chapter(6, 'Tab 3 - Uji Sentimen AI Real-Time', tab3_text)

# Tab 4 & Insight
tab4_text = """Bagian terakhir berfokus pada transparansi kinerja sistem dan kesimpulan akhir:

1. Tab Evaluasi Model (Confusion Matrix): Menampilkan rapor akurasi AI (Akurasi, Presisi, Recall, F1-Score). Metrik ini dibaca secara otomatis dari file 'eval_metrics.json' hasil pelatihan sistem, memastikan bahwa angka yang tampil adalah valid.
2. Ringkasan Fakta Otomatis: Sistem membaca data yang difilter dan menuliskan kesimpulan teks otomatis (misal: "Pelabuhan Batam Centre mencatatkan keluhan tertinggi... Aspek masalah utamanya adalah Imigrasi...").
3. Contoh Ulasan: Menampilkan dua sampel ulasan keluhan asli terbaru beserta tanggal dan jumlah bintangnya.
4. Tabel Data Mentah: Tabel yang dapat di-expand (dibuka) di bagian paling bawah untuk melihat ribuan baris data asli bergaya Excel."""
pdf.print_chapter(7, 'Tab 4 & Ringkasan Eksekutif Otomatis', tab4_text)

# Simpan PDF
output_file = "Dokumentasi_Fitur_Dashboard.pdf"
pdf.output(output_file)
print(f"File PDF '{output_file}' berhasil dibuat!")