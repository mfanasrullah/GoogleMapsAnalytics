import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os
import PIL.Image as PILImage
from datetime import datetime, timedelta
import plotly.express as px
from deep_translator import GoogleTranslator
import math
import joblib
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from config import DATA_PROCESSED
from recommendation.insight import generate_insights
from aspect.aspect import AspectExtractor
from aspect.summary import AspectSummarizer

sns.set_theme(style="whitegrid")
sns.set_palette("Blues_d")

st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide", initial_sidebar_state="expanded")

responsive_css = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="metric-container"] > div > div {
        font-size: 1.6rem !important;
        word-wrap: break-word;
    }
</style>
"""
st.markdown(responsive_css, unsafe_allow_html=True)

@st.cache_resource
def init_preprocessing_tools():
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

stemmer, stopword_remover = init_preprocessing_tools()

def preprocess_text_svm(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

@st.cache_data(ttl="1d")
def load_logo():
    logo_path = 'image_0.png' 
    try:
        image = PILImage.open(logo_path)
        return image
    except FileNotFoundError:
        logo_url = "https://www.polibatam.ac.id/wp-content/uploads/2024/01/cropped-cropped-cropped-02_Logo_1_Utama_Polibatam_Horizontal@2x.png"
        return logo_url
    except Exception:
        return None

logo_polibatam = load_logo()

def parse_gmaps_time(time_str):
    if pd.isna(time_str) or str(time_str).strip() == "": 
        return datetime.now()
        
    time_str = str(time_str).lower()
    now = datetime.now()
    
    if any(x in time_str for x in ['sebulan', 'a month', '1 month']): return now - timedelta(days=30)
    if any(x in time_str for x in ['setahun', 'a year', '1 year']): return now - timedelta(days=365)
    if any(x in time_str for x in ['seminggu', 'a week', '1 week']): return now - timedelta(days=7)
    if any(x in time_str for x in ['sehari', 'a day', '1 day']): return now - timedelta(days=1)
    if any(x in time_str for x in ['sejam', 'an hour', '1 hour']): return now - timedelta(hours=1)
    if any(x in time_str for x in ['baru saja', 'just now', 'minutes']): return now
    
    num = re.findall(r'\d+', time_str)
    if not num: 
        return now
        
    num = int(num[0])
    
    if 'tahun' in time_str or 'year' in time_str: return now - timedelta(days=num*365)
    if 'bulan' in time_str or 'month' in time_str: return now - timedelta(days=num*30)
    if 'minggu' in time_str or 'week' in time_str: return now - timedelta(days=num*7)
    if 'hari' in time_str or 'day' in time_str: return now - timedelta(days=num)
    if 'jam' in time_str or 'hour' in time_str: return now - timedelta(hours=num)
    
    return now

@st.cache_data(ttl="1d") 
def load_data():
    file_path = os.path.join(DATA_PROCESSED, "final_dataset.csv")
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    df = df.rename(columns={'location': 'pelabuhan', 'text': 'review_text'})
    
    if 'rating' in df.columns:
        df['review_rating'] = df['rating'].astype(str).str.extract(r'(\d+)').astype(float)
    
    if 'time' in df.columns:
        df['tanggal'] = df['time'].apply(parse_gmaps_time)
        df['bulan_tahun'] = df['tanggal'].dt.to_period('M').astype(str)

    if 'final_text' in df.columns:
        extractor = AspectExtractor(method='rule-based')
        df = extractor.process_dataframe(df, text_column='final_text')
    
    return df

@st.cache_resource
def load_svm_model():
    try:
        svm_model = joblib.load('data/models/svm_model.pkl')
        tfidf_vectorizer = joblib.load('data/models/tfidf_vectorizer.pkl')
        return svm_model, tfidf_vectorizer
    except FileNotFoundError:
        return None, None

df_full = load_data()
svm_model, tfidf_vectorizer = load_svm_model()

if df_full is None or df_full.empty:
    st.error("Data ulasan belum tersedia. Silakan jalankan script pengumpulan data terlebih dahulu.")
    st.stop()

with st.sidebar:
    if logo_polibatam:
        if isinstance(logo_polibatam, PILImage.Image):
            logo_resized = logo_polibatam.resize((150, int(150 * logo_polibatam.height / logo_polibatam.width)))
            st.image(logo_resized, use_container_width=True)
        else:
            st.image(logo_polibatam, use_container_width=True)
    else:
        st.write("**[POLIBATAM]**")
    
    st.markdown("## Pusat Data Pelabuhan")
    
    if st.button("🔄 Segarkan Data Sekarang", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🏢 Pilih Pelabuhan")
    all_ports = df_full['pelabuhan'].unique().tolist()
    selected_ports = st.multiselect(
        "Pilih Pelabuhan:",
        options=all_ports,
        default=all_ports 
    )
    
    st.markdown("#### Pilih Rentang Tanggal")
    min_date = df_full['tanggal'].min().date()
    max_date = df_full['tanggal'].max().date()
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Mulai", value=min_date, min_value=min_date, max_value=max_date)
    with col_date2:
        end_date = st.date_input("📅 Akhir", value=max_date, min_value=min_date, max_value=max_date)
        
    if start_date > end_date:
        st.error("⚠️ Tgl Mulai tidak boleh melewati Tgl Akhir!")

    st.markdown("---")
    st.markdown("<small>Dikembangkan oleh Tim Analitik Polibatam</small>", unsafe_allow_html=True)


df_working = df_full[df_full['pelabuhan'].isin(selected_ports)]

if start_date <= end_date:
    df_working = df_working[
        (df_working['tanggal'].dt.date >= start_date) & 
        (df_working['tanggal'].dt.date <= end_date)
    ]
else:
    df_working = pd.DataFrame(columns=df_full.columns) 

st.title("Dashboard Analisis Sentimen Pelabuhan")
st.markdown("<p style='font-size: 16px; color: gray; margin-top:-15px;'>powered by Tim Analitik Polibatam</p>", unsafe_allow_html=True)
st.markdown("---")

if df_working.empty:
    st.warning("⚠️ Tidak ada data pelabuhan yang dipilih atau sesuai rentang waktu. Sesuaikan filter di sidebar.")
    st.stop()


kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

total_reviews = len(df_working)
avg_rating = df_working['review_rating'].mean() if 'review_rating' in df_working.columns else 0
ports_counted = df_working['pelabuhan'].nunique()

with kpi_col1:
    st.metric(label="Total Volume Ulasan", value=f"{total_reviews:,}")
with kpi_col2:
    st.metric(label="Rata-rata Rating (Bintang)", value=f"{avg_rating:.1f} ⭐")
with kpi_col3:
    st.metric(label="Pelabuhan Teranalisis", value=f"{ports_counted}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualisasi Data", 
    "☁️ WordCloud & Heatmap", 
    "🤖 Prediksi Sentimen (SVM)",
    "📈 Evaluasi Model"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribusi Popularitas")
        pop_df = df_working['pelabuhan'].value_counts().reset_index()
        pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
        
        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_d', ax=ax_pop) 
        ax_pop.set_xlabel("Total Ulasan (Volume)", fontsize=10)
        ax_pop.set_ylabel("", fontsize=10)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_pop, use_container_width=True)

    with col2:
        st.markdown("#### Kualitas (Rating) vs Volume")
        if 'review_rating' in df_working.columns:
            scatter_df = df_working.groupby('pelabuhan').agg(
                Rata_Rating=('review_rating', 'mean'),
                Volume=('review_rating', 'count')
            ).reset_index()
            
            fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=scatter_df, x='Volume', y='Rata_Rating', hue='pelabuhan', s=200, palette='deep', ax=ax_scat)
            ax_scat.set_xlabel("Volume (Jumlah Ulasan)", fontsize=10)
            ax_scat.set_ylabel("Kualitas (Rata-rata Rating)", fontsize=10)
            ax_scat.set_ylim(1, 5.5) 
            ax_scat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Pelabuhan')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig_scat, use_container_width=True)

    st.markdown("---")
    
    st.markdown("#### Tren Volume Ulasan per Bulan")
    if 'bulan_tahun' in df_working.columns:
        trend_df = df_working.groupby(['bulan_tahun', 'pelabuhan']).size().reset_index(name='Jumlah')
        trend_df = trend_df.sort_values('bulan_tahun')
        
        fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=trend_df, x='bulan_tahun', y='Jumlah', hue='pelabuhan', marker='o', palette='muted', ax=ax_trend)
        ax_trend.tick_params(axis='x', rotation=45, labelsize=9) 
        ax_trend.set_xlabel("Periode (Bulan)", fontsize=10)
        ax_trend.set_ylabel("Volume Ulasan", fontsize=10)
        ax_trend.grid(True, linestyle='--', alpha=0.6)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_trend, use_container_width=True)
        
    st.markdown("---")
    
    st.markdown("#### Analisis Aspek Keluhan & Pujian")
    if 'aspects' in df_working.columns and 'sentiment' in df_working.columns:
        summarizer = AspectSummarizer()
        df_aspect_summary = summarizer.get_aspect_sentiment_summary(
            df_working, aspect_col='aspects', sentiment_col='sentiment', location_col='pelabuhan'
        )
        
        label_mapping = {
            "LABEL_0": "POSITIF", "LABEL_1": "NETRAL", "LABEL_2": "NEGATIF",
            "POSITIVE": "POSITIF", "NEUTRAL": "NETRAL", "NEGATIVE": "NEGATIF"
        }
        df_aspect_summary['sentiment'] = df_aspect_summary['sentiment'].replace(label_mapping)

        urutan_sentimen = ["NEGATIF", "NETRAL", "POSITIF"]
        
        jml_pelabuhan_unik = df_aspect_summary['pelabuhan'].nunique()
        baris_dibutuhkan = math.ceil(jml_pelabuhan_unik / 2) 
        dynamic_height = max(500, baris_dibutuhkan * 400)
        
        fig_aspect = px.bar(
            df_aspect_summary, 
            x='aspects', 
            y='count', 
            color='sentiment', 
            facet_col='pelabuhan', 
            facet_col_wrap=2,
            facet_row_spacing=0.15,   
            facet_col_spacing=0.08, 
            category_orders={"sentiment": urutan_sentimen}, 
            color_discrete_map={
                "POSITIF": "#2E7D32", 
                "NETRAL": "#B0BEC5",  
                "NEGATIF": "#E53935"  
            },
            labels={'aspects': '', 'count': 'Jumlah Ulasan', 'sentiment': 'Sentimen:'}
        )
        
        fig_aspect.update_layout(
            height=dynamic_height,             
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12, color="#424242"),
            margin=dict(t=80, b=50), 
            legend=dict(
                orientation="h", 
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title_font=dict(size=1) 
            )
        )
        
        fig_aspect.update_xaxes(matches=None, showticklabels=True, tickangle=-45, showgrid=False, title_text='')
        fig_aspect.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE', title_text='')
        fig_aspect.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=14)))
        
        st.plotly_chart(fig_aspect, use_container_width=True)

    # =====================================================================
    # FITUR BARU: FILTER & TREN ASPEK DARI WAKTU KE WAKTU (PREDIKTIF)
    # =====================================================================
    st.markdown("---")
    st.markdown("#### 📈 Tren Aspek dari Waktu ke Waktu (Analisis Prediktif)")
    st.write("Pantau kapan suatu aspek sering dibicarakan untuk memprediksi potensi masalah di masa depan berdasarkan tren bulan-bulan sebelumnya.")

    if 'aspects' in df_working.columns and 'bulan_tahun' in df_working.columns:
        
        # Persiapan data: Memastikan aspek difilter dengan baik
        df_trend_base = df_working.copy()
        
        # Jika aspek berbentuk list di dalam dataframe, kita explode agar bisa dihitung per aspek
        if not df_trend_base.empty and isinstance(df_trend_base['aspects'].iloc[0], list):
            df_trend_base = df_trend_base.explode('aspects')
            
        unique_aspects = [asp for asp in df_trend_base['aspects'].unique() if pd.notna(asp) and str(asp).strip() != ""]
        
        if len(unique_aspects) > 0:
            col_filter1, col_filter2 = st.columns([2, 1])
            
            with col_filter1:
                # Filter untuk memilih Aspek
                selected_trend_aspects = st.multiselect(
                    "🔍 Filter Aspek (Bisa pilih lebih dari satu):",
                    options=unique_aspects,
                    default=unique_aspects[:3] if len(unique_aspects) >= 3 else unique_aspects
                )
            
            with col_filter2:
                # Filter untuk fokus pada Sentimen (Berguna untuk memprediksi Keluhan)
                sentimen_fokus = st.radio(
                    "🎯 Fokus Analisis:",
                    ["Semua Ulasan", "Khusus Keluhan (Negatif)"],
                    horizontal=False
                )

            if selected_trend_aspects:
                # Memfilter data berdasarkan aspek yang dipilih
                df_trend_aspect = df_trend_base[df_trend_base['aspects'].isin(selected_trend_aspects)]
                
                # Memfilter hanya keluhan jika dipilih (berdasarkan rating 1 dan 2)
                if sentimen_fokus == "Khusus Keluhan (Negatif)" and 'review_rating' in df_trend_aspect.columns:
                    df_trend_aspect = df_trend_aspect[df_trend_aspect['review_rating'] <= 2]

                if not df_trend_aspect.empty:
                    # Mengelompokkan data berdasarkan Bulan, Pelabuhan, dan Aspek
                    trend_data = df_trend_aspect.groupby(['bulan_tahun', 'pelabuhan', 'aspects']).size().reset_index(name='Frekuensi')
                    trend_data = trend_data.sort_values('bulan_tahun')

                    # Tampilkan keterangan fokus sentimen di luar grafik (pakai Streamlit) agar lebih rapi
                    st.markdown(f"<p style='text-align: center; color: gray;'>Data yang ditampilkan: <b>{sentimen_fokus}</b></p>", unsafe_allow_html=True)

                    # Membuat Line Chart Plotly
                    fig_aspect_trend = px.line(
                        trend_data,
                        x='bulan_tahun',
                        y='Frekuensi',
                        color='aspects',
                        facet_col='pelabuhan',
                        facet_col_wrap=2,
                        markers=True,
                        line_shape='spline', # Membuat garis melengkung/halus
                        labels={'bulan_tahun': 'Bulan', 'Frekuensi': 'Jumlah Kemunculan'}
                    )

                    # Mempercantik tampilan layout Plotly
                    tinggi_grafik = max(400, math.ceil(df_trend_aspect['pelabuhan'].nunique() / 2) * 350)
                    fig_aspect_trend.update_layout(
                        height=tinggi_grafik,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified", # Menyatukan tooltip saat di hover per bulan
                        margin=dict(t=40, b=100), # Beri ruang lebih di atas dan bawah
                        legend=dict(
                            orientation="h", 
                            yanchor="top",
                            y=-0.15,           # Posisi di bawah grafik
                            xanchor="center", 
                            x=0.5,
                            title_text=""      # Menghilangkan teks "Aspek"
                        )
                    )
                    
                    fig_aspect_trend.update_xaxes(showgrid=False, tickangle=-45, title_text='')
                    fig_aspect_trend.update_yaxes(showgrid=True, gridcolor='#EEEEEE', title_text='Jumlah')
                    fig_aspect_trend.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))

                    st.plotly_chart(fig_aspect_trend, use_container_width=True)
                else:
                    st.info(f"Tidak ada data untuk aspek yang dipilih pada filter **{sentimen_fokus}** di rentang waktu ini.")
            else:
                st.warning("⚠️ Silakan pilih minimal satu aspek pada filter di atas.")
        else:
            st.warning("Data aspek belum diekstraksi. Pastikan model aspect extractor berjalan dengan benar.")

with tab2:
    col_wc, col_hm = st.columns(2)
    
    with col_wc:
        st.markdown("#### Visualisasi WordCloud")
        if 'final_text' in df_working.columns or 'review_text' in df_working.columns:
            teks_kolom = 'final_text' if 'final_text' in df_working.columns else 'review_text'
            semua_teks = " ".join(df_working[teks_kolom].dropna().astype(str))
            
            if semua_teks.strip(): 
                wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='Blues').generate(semua_teks)
                fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.info("Tidak ada data teks ulasan yang cukup untuk membuat WordCloud.")

    with col_hm:
        st.markdown("#### Heatmap Keluhan Konsumen")
        if 'bulan_tahun' in df_working.columns and 'review_rating' in df_working.columns:
            df_negatif = df_working[df_working['review_rating'] <= 2]
            
            if not df_negatif.empty:
                pivot_keluhan = df_negatif.pivot_table(
                    index='pelabuhan', 
                    columns='bulan_tahun', 
                    values='review_rating', 
                    aggfunc='count', 
                    fill_value=0
                )
                
                fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                sns.heatmap(pivot_keluhan, cmap='Reds', annot=True, fmt='d', linewidths=.5, ax=ax_hm, annot_kws={"size": 10})
                ax_hm.set_xlabel("Periode (Bulan)", fontsize=9)
                ax_hm.set_ylabel("", fontsize=9)
                sns.despine(left=True, bottom=True)
                st.pyplot(fig_hm, use_container_width=True)
            else:
                st.success("Luar biasa! Tidak ada ulasan negatif (Rating 1 & 2) yang ditemukan dalam rentang waktu terfilter.")

with tab3:
    st.header("Sistem Uji Sentimen Real-Time")
    st.markdown("Ketik ulasan di bawah ini untuk melihat bagaimana **Support Vector Machine (SVM) dan TF-IDF** memprediksi sentimen teks secara instan berdasarkan data latih.")
    
    if svm_model is None or tfidf_vectorizer is None:
        st.error("⚠️ Model SVM (`svm_model.pkl`) atau TF-IDF Vectorizer (`tfidf_vectorizer.pkl`) belum tersedia di folder `data/models/`. Silakan jalankan script pelatihan model terlebih dahulu.")
    else:
        user_input = st.text_area("Ketik ulasan terkait layanan pelabuhan (maks. 500 kata):", height=120)
        
        if st.button("Analisis Ulasan (SVM)", type="primary", use_container_width=True):
            if user_input:
                with st.spinner('Memproses teks (Preprocessing & TF-IDF)...'):
                    try:
                        teks_terjemahan = GoogleTranslator(source='auto', target='id').translate(user_input)
                    except:
                        teks_terjemahan = user_input 
                        
                    teks_bersih = preprocess_text_svm(teks_terjemahan)
                    
                    vektor_teks = tfidf_vectorizer.transform([teks_bersih])
                    
                    raw_prediksi = svm_model.predict(vektor_teks)[0]
                    
                    if hasattr(svm_model, "predict_proba"):
                        probabilitas = svm_model.predict_proba(vektor_teks)[0]
                        kelas_model = svm_model.classes_
                        
                        prob_pos, prob_neu, prob_neg = 0.0, 0.0, 0.0
                        for i, kls in enumerate(kelas_model):
                            if kls.upper() == "POSITIF" or kls == 1 or kls == 2: prob_pos = probabilitas[i]
                            elif kls.upper() == "NEGATIF" or kls == -1 or kls == 0: prob_neg = probabilitas[i]
                            else: prob_neu = probabilitas[i]
                    else:
                        prob_pos, prob_neu, prob_neg = (1.0, 0.0, 0.0) if raw_prediksi == "POSITIF" else (0.0, 0.0, 1.0) if raw_prediksi == "NEGATIF" else (0.0, 1.0, 0.0)

                    prediksi = str(raw_prediksi).upper()
                    if prediksi in ["0", "-1"]: prediksi = "NEGATIF"
                    elif prediksi in ["1"]: prediksi = "NETRAL"
                    elif prediksi in ["2"]: prediksi = "POSITIF"
                    
                    result_col1, result_col2 = st.columns(2)
                    warna = "green" if prediksi == "POSITIF" else "red" if prediksi == "NEGATIF" else "gray"
                    
                    with result_col1:
                        st.markdown(f"<div style='border: 1px solid lightgray; padding: 10px; border-radius: 5px; text-align: center;'>Hasil Prediksi AI:<br><b style='color:{warna}; font-size: 24px;'>{prediksi}</b></div>", unsafe_allow_html=True)
                        st.markdown(f"<br><small style='color:gray;'><b>Teks yang masuk ke model (setelah Sastrawi):</b><br> '{teks_bersih}'</small>", unsafe_allow_html=True)
                    
                    with result_col2:
                        st.write("**Tingkat Keyakinan SVM (Probabilitas):**")
                        st.progress(float(prob_pos), text=f"Positif: {prob_pos:.1%}")
                        st.progress(float(prob_neu), text=f"Netral: {prob_neu:.1%}") 
                        st.progress(float(prob_neg), text=f"Negatif: {prob_neg:.1%}")
                        if not hasattr(svm_model, "predict_proba"):
                            st.warning("Model SVM Anda saat ini tidak dikonfigurasi untuk mengeluarkan probabilitas (probability=False saat training).")

with tab4:
    st.header("Evaluasi Kinerja Model SVM")
    st.markdown("Bagian ini menampilkan metrik performa model Support Vector Machine.")
    
    metrik_col1, metrik_col2, metrik_col3, metrik_col4 = st.columns(4)

    akurasi_val = 0.72
    presisi_val = 0.76
    recall_val = 0.72
    f1_val = 0.73
    
    with metrik_col1: st.metric(label="Accuracy", value=f"{akurasi_val:.1%}")
    with metrik_col2: st.metric(label="Presisi Precision", value=f"{presisi_val:.1%}")
    with metrik_col3: st.metric(label="Recall", value=f"{recall_val:.1%}")
    with metrik_col4: st.metric(label="F1-Score", value=f"{f1_val:.1%}")
    
    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    st.info("Visualisasi ini menunjukan kemampuan model dalam membedakan setiap kelas (Positif, Netral, Negatif). Sumbu Y adalah kelas asli (Actual), dan Sumbu X adalah tebakan model (Predicted).")
    
    # DATA CONFUSION MATRIX YANG SESUAI DENGAN EVALUASI RIIL (Total 486 data uji)
    data_cm = np.array([
        [38, 11, 12],
        [10, 23, 28],
        [26, 50, 288]
    ])
    labels = ['Negatif', 'Netral', 'Positif']
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(data_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    st.pyplot(fig_cm)

st.markdown("---")
st.subheader("💡 Insights & Rekomendasi ")
if 'sentiment' in df_working.columns:
    insights = generate_insights(df_working)
    for i, insight in enumerate(insights, 1):
        st.info(f"**{i}.** {insight}")

with st.expander("Lihat Data Ulasan Mentah (Tabel)"):
    cols_to_show = ['pelabuhan', 'tanggal', 'review_text', 'review_rating']
    available_cols = [c for c in cols_to_show if c in df_working.columns]
    
    if 'aspects' in df_working.columns:
        available_cols.append('aspects')
        
    st.dataframe(df_working[available_cols], use_container_width=True)
