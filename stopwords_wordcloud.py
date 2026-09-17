# -*- coding: utf-8 -*-
"""
Stopword khusus untuk WordCloud - Dashboard Analisis Sentimen Pelabuhan.

File ini adalah SATU-SATUNYA sumber (single source of truth) untuk daftar
stopword WordCloud, dipakai bersama oleh:
  - preprocessing.py  (saat membangun kolom 'wordcloud_text' di dataset)
  - app.py            (sebagai filter tambahan saat render WordCloud di Streamlit)

PENTING: daftar ini SENGAJA dipisah dari stopword yang dipakai untuk melatih/
menjalankan model SVM (final_text). Mengubah daftar ini TIDAK mempengaruhi
model SVM yang sudah dilatih - aman untuk terus disesuaikan kapan saja.

Filosofi penyusunan (disesuaikan dengan judul riset "Analisis Sentimen
Pelabuhan"):
  - DIBUANG : kata fungsi/tata bahasa (ID & EN) - kata ganti, kata depan,
              kata sambung, kata tanya, partikel, angka/urutan, dsb.
  - DIBUANG : nama lokasi/instansi/merek (Batam, Singapura, Sekupang, dll)
              dan kata logistik/prosedural (tiket, imigrasi, antrian, gate,
              boarding, dll) karena topik ini sudah terwakili di visualisasi
              Aspek Keluhan & Pujian terpisah.
  - DIPERTAHANKAN : kata sifat/kualitas & sentimen dalam ID maupun EN
              (bagus, nyaman, bersih, ramah, rude, corruption, crowded,
              comfortable, dst) karena inilah inti dari WordCloud sentimen.
"""

WORDCLOUD_STOPWORDS = frozenset({
    'a', 'about', 'above', 'actually', 'ada', 'adalah', 'after', 'again', 'against', 'agak',
    'agar', 'air', 'airport', 'aja', 'akan', 'akhirnya', 'akibat', 'aku', 'alasan', 'all',
    'allowed', 'already', 'also', 'always', 'am', 'ambil', 'an', 'anak', 'and', 'anda',
    'angkutan', 'another', 'antar', 'antre', 'antrean', 'antri', 'antrian', 'any', 'anyone',
    'apa', 'apakah', 'arah', 'are', 'area', 'around', 'arrival', 'arrived', 'as', 'ask',
    'asked', 'at', 'atas', 'atau', 'atm', 'autogate', 'available', 'back', 'bagaimana',
    'baggage', 'bagi', 'bahkan', 'bahwa', 'bandara', 'banget', 'bangunan', 'banyak', 'barang',
    'basic', 'basically', 'batam', 'bawa', 'bay', 'bayar', 'be', 'bea', 'beberapa', 'because',
    'been', 'before', 'begini', 'begitu', 'being', 'beli', 'beliau', 'below', 'belum', 'berada',
    'berangkat', 'berbagai', 'beri', 'berjalan', 'bertanya', 'best', 'better', 'between', 'bgt',
    'biar', 'biasa', 'biasanya', 'bilang', 'bintan', 'bisa', 'bit', 'board', 'boarding', 'boat',
    'boleh', 'both', 'bought', 'bridge', 'buat', 'building', 'bukan', 'bus', 'busy', 'but',
    'buy', 'by', 'cafe', 'came', 'can', 'cannot', 'cant', 'cari', 'cash', 'center', 'centre',
    'change', 'changed', 'changer', 'changi', 'check', 'check-in', 'checked', 'checkin',
    'checking', 'city', 'clearance', 'close', 'come', 'coming', 'could', 'couldnt', 'counter',
    'counters', 'country', 'covid', 'crowd', 'cukai', 'cukup', 'cuma', 'customs', 'd', 'dalam',
    'dan', 'dapat', 'dari', 'datang', 'day', 'days', 'definitely', 'deh', 'dengan', 'departure',
    'dermaga', 'di', 'dia', 'did', 'didnt', 'disini', 'dll', 'do', 'does', 'doesnt', 'doing',
    'domestic', 'domestik', 'don', 'dong', 'dont', 'down', 'driver', 'drivers', 'dst', 'dua',
    'duduk', 'due', 'dulu', 'dumai', 'during', 'each', 'eh', 'empat', 'enough', 'enter',
    'entering', 'entrance', 'entry', 'eskalator', 'even', 'ever', 'every', 'everybody',
    'everyone', 'everything', 'experience', 'far', 'feel', 'felt', 'feri', 'ferry', 'fery',
    'few', 'find', 'first', 'floor', 'food court', 'for', 'free', 'from', 'front', 'further',
    'gate', 'gedung', 'gerbang', 'get', 'getting', 'gimana', 'gini', 'gitu', 'give', 'given',
    'go', 'going', 'gone', 'got', 'grab', 'guna', 'guy', 'had', 'hal', 'hampir', 'hanya',
    'happened', 'harbor', 'harbour', 'harbourfront', 'hari', 'harus', 'has', 'hati', 'have',
    'he', 'help', 'helped', 'her', 'here', 'hers', 'hes', 'him', 'hingga', 'his', 'holiday',
    'honestly', 'hot', 'hotel', 'hour', 'hours', 'how', 'hubung', 'i', 'ia', 'ialah', 'ibu',
    'idr', 'if', 'im', 'imigrasi', 'immigration', 'in', 'indonesia', 'indonesian', 'ingin',
    'ini', 'inside', 'instead', 'internasional', 'international', 'into', 'is', 'island',
    'islands', 'isnt', 'it', 'its', 'itu', 'ive', 'iya', 'jadi', 'jadwal', 'jalan', 'jalur',
    'jam', 'jangan', 'jembatan', 'jika', 'job', 'johor', 'juga', 'just', 'kah', 'kalau', 'kali',
    'kalo', 'kami', 'kamu', 'kan', 'kantor', 'kapal', 'kapalnya', 'kapan', 'karena', 'karimun',
    'kata', 'ke', 'keberangkatan', 'kedatangan', 'kedua', 'keluar', 'kembali', 'kemudian',
    'kenapa', 'kendaraan', 'kepada', 'kepri', 'kepulauan', 'kesini', 'ketiga', 'ketika', 'kind',
    'kita', 'klo', 'know', 'kok', 'konter', 'kota', 'krn', 'kurang', 'lady', 'lagi', 'lah',
    'lain', 'lainnya', 'laku', 'lakukan', 'lalu', 'lama', 'langsung', 'lantai', 'last', 'laut',
    'lebih', 'left', 'less', 'let', 'level', 'lewat', 'lho', 'lift', 'lihat', 'like', 'lima',
    'line', 'literally', 'little', 'll', 'local', 'located', 'loh', 'lokasi', 'loket', 'long',
    'look', 'looks', 'lot', 'lots', 'luar', 'luggage', 'lumayan', 'm', 'macam', 'made', 'main',
    'make', 'makin', 'making', 'mal', 'malah', 'malam', 'malaysia', 'mall', 'mana',
    'management', 'many', 'masih', 'masing', 'masuk', 'mau', 'maupun', 'maybe', 'me', 'mega',
    'mega mall', 'megamall', 'melakukan', 'melalui', 'melihat', 'memang', 'membayar', 'membeli',
    'memberikan', 'membuat', 'memiliki', 'mengambil', 'mengapa', 'mengatakan', 'menggunakan',
    'menit', 'menjadi', 'menuju', 'menunggu', 'menyeberang', 'merah', 'merasa', 'mereka',
    'merupakan', 'meski', 'meskipun', 'might', 'miliki', 'min', 'minta', 'minute', 'minutes',
    'mobil', 'mohon', 'money', 'money changer', 'more', 'most', 'motor', 'much', 'mulai',
    'mungkin', 'must', 'my', 'nah', 'naik', 'namun', 'nan', 'near', 'need', 'needed', 'needs',
    'negara', 'never', 'new', 'next', 'nih', 'no', 'nongsa', 'nongsapura', 'nor', 'not',
    'nothing', 'now', 'nya', 'nyebrang', 'of', 'off', 'office', 'officer', 'officers', 'often',
    'oh', 'ojek', 'ok', 'okay', 'oke', 'old', 'oleh', 'on', 'once', 'one', 'only', 'open',
    'operator', 'option', 'or', 'orang', 'orang-orang', 'order', 'org', 'other', 'our', 'ours',
    'out', 'outside', 'over', 'own', 'pada', 'padahal', 'pagi', 'paid', 'pakai', 'pake',
    'paling', 'panjang', 'para', 'parking', 'parkir', 'parkiran', 'parkirnya', 'pas', 'paspor',
    'pass', 'passenger', 'passengers', 'passport', 'pasti', 'pay', 'pelabuhan', 'pelabuhannya',
    'pembangunan', 'pengalaman', 'pengunjung', 'penumpang', 'penyeberangan', 'penyebrangan',
    'people', 'per', 'pergi', 'perhaps', 'perjalanan', 'perlu', 'pernah', 'person', 'pertama',
    'petugas', 'petunjuk', 'phone', 'pick', 'pilihan', 'pinang', 'pintu', 'place', 'please',
    'point', 'points', 'port', 'porter', 'ports', 'probably', 'process', 'processing', 'proses',
    'public', 'pukul', 'pulang', 'pulau', 'pun', 'punggur', 'punya', 'pusat', 'queue', 'quite',
    're', 'reach', 'really', 'renovasi', 'resort', 'restaurant', 'restoran', 'return',
    'returned', 'riau', 'right', 'room', 'roro', 'route', 'rp', 'ruang', 'ruangan', 'rupiah',
    'rute', 's', 'saat', 'said', 'saja', 'salah', 'sama', 'sambil', 'same', 'sampai', 'sana',
    'sangat', 'satu', 'saw', 'saya', 'sayang', 'sdh', 'sea', 'sebab', 'sebagai', 'sebelum',
    'sebelumnya', 'secara', 'second', 'sedang', 'sedangkan', 'sedikit', 'see', 'seem', 'seems',
    'seen', 'segala', 'sehingga', 'sejak', 'sekali', 'sekarang', 'sekitar', 'sekupang',
    'selain', 'selalu', 'selama', 'selesai', 'seluruh', 'semakin', 'sementara', 'semoga',
    'semua', 'sendiri', 'sendirinya', 'seperti', 'seriously', 'serta', 'service point',
    'services', 'setelah', 'setiap', 'sg', 'sgd', 'shall', 'she', 'shes', 'ship', 'shop',
    'shopping', 'shops', 'should', 'shouldnt', 'siang', 'siapa', 'sih', 'since', 'singapore',
    'singapura', 'sini', 'sistem', 'situ', 'snack', 'so', 'some', 'someone', 'something',
    'sometimes', 'sopir', 'sore', 'sort', 'souvenir', 'speed', 'staf', 'staff', 'still', 'such',
    'sudah', 'supaya', 'super', 'sure', 'system', 't', 'tahu', 'tahun', 'tak', 'take', 'taken',
    'taksi', 'tanah', 'tanjung', 'tanjungpinang', 'tanpa', 'tapi', 'taxi', 'taxis', 'telaga',
    'tempat', 'tempatnya', 'tentu', 'terasa', 'terdapat', 'terjadi', 'terlalu', 'terletak',
    'terlihat', 'terminal', 'terminalnya', 'tersedia', 'terus', 'tetap', 'tetapi', 'tg', 'than',
    'thank', 'thanks', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'then', 'there',
    'they', 'theyre', 'thing', 'things', 'this', 'through', 'tiap', 'tiba', 'ticket', 'tickets',
    'tidak', 'tiga', 'tiket', 'tiketnya', 'time', 'times', 'tinggal', 'to', 'today', 'toilet',
    'toko', 'told', 'tolong', 'too', 'took', 'tourist', 'tourists', 'transit', 'transportation',
    'travel', 'trip', 'tuh', 'tuju', 'tujuan', 'tunggu', 'turun', 'two', 'uang', 'uban', 'udah',
    'umum', 'under', 'until', 'untuk', 'up', 'us', 'use', 'used', 'using', 'usually', 'utama',
    'utk', 've', 'very', 'via', 'vip', 'visa', 'visit', 'wait', 'waiting', 'wajib', 'waktu',
    'walau', 'walaupun', 'want', 'wanted', 'warung', 'was', 'wasnt', 'way', 'ways', 'we',
    'weekend', 'well', 'went', 'were', 'werent', 'what', 'whats', 'when', 'where', 'which',
    'while', 'who', 'why', 'will', 'with', 'without', 'wont', 'worse', 'worst', 'would',
    'wouldnt', 'ya', 'yah', 'yaitu', 'yakni', 'yang', 'year', 'years', 'yesterday', 'yet', 'yg',
    'you', 'your', 'youre', 'yours'
})
