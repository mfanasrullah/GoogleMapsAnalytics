import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/processed/final_dataset.csv'
MODEL_DIR = 'data/models'

print("="*50)
print("🚀 MEMULAI PELATIHAN MODEL SVM & TF-IDF (GROUND TRUTH) 🚀")
print("="*50)

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"❌ File {DATA_PATH} tidak ditemukan!")
    exit()

print("1. Memuat dataset final_dataset.csv...")
df = pd.read_csv(DATA_PATH)

print("2. Mengekstrak Ground Truth dari Rating Bintang...")
df['rating_num'] = df['rating'].astype(str).str.extract(r'(\d+)').astype(float)
        
def map_sentiment(r):
    if r >= 4: return 'POSITIF'
    elif r <= 2: return 'NEGATIF'
    else: return 'NETRAL'
    
df['sentiment_actual'] = df['rating_num'].apply(map_sentiment)
df = df.dropna(subset=['final_text', 'sentiment_actual'])

print("3. Membagi data uji dan latih (80:20)...")
X = df['final_text']
y = df['sentiment_actual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("4. Melakukan pembobotan TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("5. Melatih model Support Vector Machine (SVM)...")
svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_vec, y_train)

print("\n" + "="*50)
print("📊 HASIL EVALUASI MODEL SVM 📊")
print("="*50)
y_pred = svm_model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ==============================================================
# [BARU]: MENGHITUNG DAN MENYIMPAN METRIK UNTUK STREAMLIT
# ==============================================================
labels = ['NEGATIF', 'NETRAL', 'POSITIF']
cm = confusion_matrix(y_test, y_pred, labels=labels)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    # Menggunakan weighted average untuk multiclass
    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    'confusion_matrix': cm.tolist(), # Konversi array NumPy ke list agar bisa di-JSON-kan
    'labels': labels
}

# Menyimpan metrik ke file JSON
metrics_path = os.path.join(MODEL_DIR, 'eval_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f)

print(f"\nMatriks Kebingungan (Confusion Matrix) disimpan ke {metrics_path}")

print("Menyimpan model ke dalam folder data/models/ ...")
joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

print("✅ Selesai! Model SVM kini dilatih dengan kebenaran aktual pengguna.")