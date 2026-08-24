# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Menggunakan Dictionary untuk menyimpan Direct URL
# Parameter hl=en diubah menjadi hl=id agar UI Google Maps konsisten berbahasa Indonesia
TARGET_LOCATIONS = {
    "Pemanasan Browser": "https://www.google.com/maps/place/Politeknik+Negeri+Batam",
    # --- BATAM CENTRE DIPECAH JADI 2 LINK ---
    "Batam Centre Ferry Terminal (A)": "https://www.google.com/maps/place/Batam+Centre+Point+International+Ferry+Terminal/data=!4m7!3m6!1s0x31d9891d13f27f6b:0xadc0424017784eed!8m2!3d1.1307223!4d104.0551943!16s%2Fg%2F1pztqj9mj!19sChIJa3_yEx2J2TER7U54F0BCwK0?authuser=0&hl=id&rclk=1",

    "Batam Centre Ferry Terminal (B)": "https://www.google.com/maps/place/Batam+Center+International+Ferry+Terminal/data=!4m7!3m6!1s0x31d988fd04f61e15:0x6ded49c5814062a5!8m2!3d1.130607!4d104.0553512!16s%2Fg%2F11f1222m2p!19sChIJFR72BP2I2TERpWJAgcVJ7W0?authuser=0&hl=id&rclk=1",

    # ----------------------------------------
    "Harbour Bay Ferry Terminal": "https://www.google.com/maps/place/Harbour+Bay+Ferry/data=!4m7!3m6!1s0x31d98a235c935fb5:0x1babbd4c3e3fb7df!8m2!3d1.1533325!4d103.9968005!16s%2Fg%2F1pt_822vb!19sChIJtV-TXCOK2TER37c_Pky9qxs?authuser=0&hl=id&rclk=1",

    "Sekupang Ferry Terminal": "https://www.google.com/maps/place/Sekupang+Ferry+Terminal/data=!4m7!3m6!1s0x31d98b3a0508aa89:0xad56350251fd4b37!8m2!3d1.125701!4d103.9265659!16s%2Fg%2F1hm3c98jb!19sChIJiaoIBTqL2TERN0v9UQI1Vq0?authuser=0&hl=en&rclk=1",
    #"https://www.google.com/maps/place/SEKUPANG+INTERNATIONAL+FERRY+TERMINAL/data=!4m7!3m6!1s0xa925df7bfe758abb:0x20faa8488771bd65!8m2!3d1.1250921!4d103.925189!16s%2Fg%2F11ytq0szhl!19sChIJu4p1_nvfJakRZb1xh0io-iA?authuser=0&hl=id&rclk=1",

    "Telaga Punggur Ferry Terminal": "https://www.google.com/maps/place/Telaga+Punggur+Ferry+Terminal/data=!4m7!3m6!1s0x31d985307942cd99:0x455a41a413648f4a!8m2!3d1.0350946!4d104.133273!16s%2Fg%2F11c1qxnrl4!19sChIJmc1CeTCF2TERSo9kE6RBWkU?authuser=0&hl=id&rclk=1",
    "Nongsapura Ferry Terminal": "https://www.google.com/maps/place/Nongsapura+Ferry+Terminal/data=!4m7!3m6!1s0x31da27e38c6ace1d:0x3dd2efb10180b639!8m2!3d1.188648!4d104.094611!16s%2Fg%2F11cffsrz7!19sChIJHc5qjOMn2jERObaAAbHv0j0?authuser=0&hl=en&rclk=1"
}

SENTIMENT_MODEL = "mdhugol/indonesia-bert-sentiment-classification"