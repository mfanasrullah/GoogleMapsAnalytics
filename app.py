# app.py
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

from config import DATA_PROCESSED
from recommendation.insight import generate_insights
from aspect.aspect import AspectExtractor
from aspect.summary import AspectSummarizer
from sentiment.indobert import SentimentAnalyzer


# KONFIGURASI TEMA & PALETTE WARNA CORPORATE
sns.set_theme(style="whitegrid")
sns.set_palette("Blues_d")


# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analisis Pelabuhan", layout="wide")

# Menyembunyikan header, menu, dan footer default Streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data(ttl="1d")
def load_logo():
    logo_path = 'image_0.png' 
    try:
        image = PILImage.open(logo_path)
        return image
    except FileNotFoundError:
        logo_url = "https://www.polibatam.ac.id/wp-content/uploads/2024/01/cropped-cropped-cropped-02_Logo_1_Utama_Polibatam_Horizontal@2x.png"
        return logo_url
    except Exception as e:
        return None

logo_polibatam = load_logo()


# FUNGSI PARSING WAKTU DUA BAHASA (ID & EN)
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


#  FUNGSI UNTUK MEMUAT DATA & MODEL
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
def load_model():
    return SentimentAnalyzer()

df_full = load_data()
model = load_model()

if df_full is None or df_full.empty:
    st.error("Data ulasan belum tersedia. Silakan jalankan `python main.py` terlebih dahulu untuk melakukan scraping.")
    st.stop()


# SIDEBAR (FILTERS & BRANDING)
with st.sidebar:
    if logo_polibatam:
        if isinstance(logo_polibatam, PILImage.Image):
            logo_resized = logo_polibatam.resize((150, int(150 * logo_polibatam.height / logo_polibatam.width)))
            st.image(logo_resized)
        else:
            st.image(logo_polibatam, width=150)
    else:
        st.write("**[POLIBATAM]**")
    
    st.markdown("## Pusat Data Pelabuhan")
    
    if st.button("🔄 Segarkan Data Sekarang"):
        st.cache_data.clear()
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
        start_date = st.date_input("📅 Tgl Mulai", value=min_date, min_value=min_date, max_value=max_date)
    with col_date2:
        end_date = st.date_input("📅 Tgl Akhir", value=max_date, min_value=min_date, max_value=max_date)
        
    if start_date > end_date:
        st.error("⚠️ Tgl Mulai tidak boleh melewati Tgl Akhir!")

    st.markdown("---")
    st.markdown("<small>Dikembangkan oleh Tim Analitik Polibatam</small>", unsafe_allow_html=True)


# MEMPROSES DATA BERDASARKAN FILTER
df_working = df_full[df_full['pelabuhan'].isin(selected_ports)]

if start_date <= end_date:
    df_working = df_working[
        (df_working['tanggal'].dt.date >= start_date) & 
        (df_working['tanggal'].dt.date <= end_date)
    ]
else:
    df_working = pd.DataFrame(columns=df_full.columns) 

# BODY - HEADER SECTION (TITLE)
st.title("Dashboard Analisis Sentimen Pelabuhan")
st.markdown("<p style='font-size: 18px; color: gray; margin-top:-15px;'>powered by Tim Analitik Polibatam</p>", unsafe_allow_html=True)
st.markdown("---")

if df_working.empty:
    st.warning("⚠️ Tidak ada data pelabuhan yang dipilih atau sesuai rentang waktu. Sesuaikan filter di sidebar.")
    st.stop()

# BODY - KPI ROW (RINGKASAN METRIK UTAMA)
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

total_reviews = len(df_working)
avg_rating = df_working['review_rating'].mean() if 'review_rating' in df_working.columns else 0
ports_counted = df_working['pelabuhan'].nunique()

with kpi_col1:
    st.metric(label="Total Volume Ulasan", value=f"{total_reviews:,}")
with kpi_col2:
    st.metric(label="Rata-rata Rating (Bintang)", value=f"{avg_rating:.1f} ⭐")
with kpi_col3:
    st.metric(label="Jumlah Pelabuhan Teranalisis", value=f"{ports_counted}")

st.markdown("---")

# BODY - PEMBUATAN TABS
tab1, tab2, tab3 = st.tabs([
    "📊 Visualisasi Data & Tren", 
    "☁️ WordCloud & Heatmap Keluhan", 
    "🤖 Prediksi Sentimen AI"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribusi Popularitas (Volume Aktivitas)")
        pop_df = df_working['pelabuhan'].value_counts().reset_index()
        pop_df.columns = ['Pelabuhan', 'Jumlah Ulasan']
        
        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        sns.barplot(data=pop_df, x='Jumlah Ulasan', y='Pelabuhan', palette='Blues_d', ax=ax_pop) 
        ax_pop.set_xlabel("Total Ulasan (Volume)", fontsize=10)
        ax_pop.set_ylabel("", fontsize=10)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_pop)

    with col2:
        st.markdown("#### Kualitas (Rating) vs Kuantitas (Volume)")
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
            st.pyplot(fig_scat)

    st.markdown("---")
    
    # Tren Waktu
    st.markdown("#### Tren Volume Ulasan per Bulan di Setiap Pelabuhan")
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
        st.pyplot(fig_trend)
        
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
        
        fig_aspect = px.bar(
            df_aspect_summary, 
            x='aspects', 
            y='count', 
            color='sentiment', 
            facet_col='pelabuhan', 
            facet_col_wrap=3,
            facet_row_spacing=0.4,   
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
            height=900,              
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12, color="#424242"),
            margin=dict(t=100, b=50), 
            legend=dict(
                orientation="h", 
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                title_font=dict(size=1) 
            )
        )
        
        fig_aspect.update_xaxes(matches=None, showticklabels=True, tickangle=-45, showgrid=False, title_text='')
        fig_aspect.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEEEEE', title_text='')
        fig_aspect.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=14)))
        
        st.plotly_chart(fig_aspect, use_container_width=True)

with tab2:
    col_wc, col_hm = st.columns(2)
    
    with col_wc:
        st.markdown("#### Visualisasi WordCloud")
        if 'final_text' in df_working.columns or 'review_text' in df_working.columns:
            teks_kolom = 'final_text' if 'final_text' in df_working.columns else 'review_text'
            semua_teks = " ".join(df_working[teks_kolom].dropna().astype(str))
            
            if semua_teks.strip(): 
                wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(semua_teks)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
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
                st.pyplot(fig_hm)
            else:
                st.success("Luar biasa! Tidak ada ulasan negatif (Rating 1 & 2) yang ditemukan dalam rentang waktu terfilter.")

with tab3:
    st.header("Sistem Uji Sentimen Real-Time")
    st.markdown("Ketik ulasan di bawah ini untuk melihat bagaimana **Model AI IndoBERT** memprediksi sentimen teks secara instan.")
    
    if model is None:
        st.error("Model Machine Learning belum tersedia. Fitur prediksi dinonaktifkan.")
    else:
        kamus_positif = {
            'bagus', 'baik', 'cepat', 'bersih', 'ramah', 'nyaman', 'keren', 'mantap', 
            'memuaskan', 'mudah', 'rapi', 'aman', 'lancar', 'terbaik', 'puas', 'indah', 
            'luas', 'modern', 'sip', 'jos', 'lengkap', 'murah', 'tertata', 'tertib', 
            'teratur', 'strategis', 'mewah', 'megah', 'terjangkau', 'sejuk', 'elegan', 'wangi',
            'friendly', 'good', 'nice', 'clean', 'fast', 'quick', 'comfortable', 'great', 
            'awesome', 'best', 'helpful', 'excellent', 'smooth', 'clearance',
            'beautiful', 'organized', 'better', 'cool', 'easy', 'spacious', 'safe', 
            'convenient', 'efficient', 'clear', 'calm', 'perfect', 'lovely', 'cheap', 
            'top', 'affordable', 'relaxing', 'amazing'
        }
        
        kamus_negatif = {
            'buruk', 'lambat', 'kotor', 'mahal', 'antri', 'jelek', 'kecewa', 'sulit', 
            'lama', 'ribet', 'bising', 'bau', 'rusak', 'berantakan', 'parah', 'kurang', 
            'sempit', 'macet', 'panas', 'kacau', 'korupsi', 'kasar', 'jorok', 'kumuh', 
            'lelet', 'sombong', 'bertele', 'sumpek', 'pengap', 'pesing', 'jutek', 'kusam',
            'bad', 'slow', 'dirty', 'expensive', 'crowded', 'queue', 'disappointed', 
            'hard', 'difficult', 'noisy', 'smelly', 'broken', 'messy', 'worst', 'hot',
            'terrible', 'sad', 'poor', 'disgusting', 'rude', 'useless', 'chaos', 'corrupt', 
            'unpleasant', 'chaotic', 'awful', 'horrible', 'scam', 'unfriendly', 'unorganized', 
            'stinky', 'trash'
        }
        
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text.strip()
        
        user_input = st.text_area("Ketik ulasan terkait layanan pelabuhan (maks. 500 kata):", height=120)
        
        if st.button("Analisis Ulasan", type="primary"):
            if user_input:
                
                with st.spinner('Menerjemahkan dan menganalisis ulasan...'):
                    try:
                        teks_terjemahan = GoogleTranslator(source='auto', target='id').translate(user_input)
                    except:
                        teks_terjemahan = user_input 
                        
                    teks_bersih_ai = str(teks_terjemahan).lower()
                    teks_bersih_ai = re.sub(r'[^a-z\s]', '', teks_bersih_ai).strip()
                    
                    teks_bersih_kamus = str(user_input).lower()
                    teks_bersih_kamus = re.sub(r'[^a-z\s]', '', teks_bersih_kamus).strip()
                    
                    hasil_prediksi = model.nlp(teks_bersih_ai[:512])[0]
                    raw_label = hasil_prediksi['label']
                    skor = hasil_prediksi['score']
                
                label_mapping_ai = {
                    "LABEL_0": "POSITIF", 
                    "LABEL_1": "NETRAL", 
                    "LABEL_2": "NEGATIF"
                }
                prediksi = label_mapping_ai.get(raw_label, raw_label)
                
                if prediksi == "POSITIVE": prediksi = "POSITIF"
                if prediksi == "NEGATIVE": prediksi = "NEGATIF"
                if prediksi == "NEUTRAL": prediksi = "NETRAL"
                
                result_col1, result_col2 = st.columns(2)
                warna = "green" if prediksi == "POSITIF" else "red" if prediksi == "NEGATIF" else "gray"
                
                with result_col1:
                    st.markdown(f"<div style='border: 1px solid lightgray; padding: 10px; border-radius: 5px;'>Hasil Prediksi AI:<br><b style='color:{warna}; font-size: 24px;'>{prediksi.upper()}</b></div>", unsafe_allow_html=True)
                
                with result_col2:
                    st.write("**Tingkat Keyakinan (Probabilitas):**")
                    
                    if prediksi == "POSITIF":
                        prob_pos = skor
                        prob_neu = (1.0 - skor) / 2
                        prob_neg = (1.0 - skor) / 2
                    elif prediksi == "NEGATIF":
                        prob_neg = skor
                        prob_neu = (1.0 - skor) / 2
                        prob_pos = (1.0 - skor) / 2
                    else: 
                        prob_neu = skor
                        prob_pos = (1.0 - skor) / 2
                        prob_neg = (1.0 - skor) / 2
                    
                    st.progress(float(prob_pos), text=f"Positif: {prob_pos:.1%}")
                    st.progress(float(prob_neu), text=f"Netral: {prob_neu:.1%}") 
                    st.progress(float(prob_neg), text=f"Negatif: {prob_neg:.1%}")
                    
                st.markdown("---")
                st.markdown("#### 🔍 Analisis Kata Kunci dalam Ulasan")
                
                kata_dalam_teks = set(teks_bersih_kamus.split())
                kata_positif_ditemukan = kata_dalam_teks.intersection(kamus_positif)
                kata_negatif_ditemukan = kata_dalam_teks.intersection(kamus_negatif)
                
                jml_pos = len(kata_positif_ditemukan)
                jml_neg = len(kata_negatif_ditemukan)
                
                col_word1, col_word2 = st.columns(2)
                
                with col_word1:
                    if kata_positif_ditemukan:
                        kata_pos_str = ", ".join([f"`{k}`" for k in kata_positif_ditemukan])
                        st.success(f"**Kata Positif Terdeteksi ({jml_pos}):**\n\n{kata_pos_str}")
                    else:
                        st.info("**Kata Positif Terdeteksi:**\n\n0 kata.")
                        
                with col_word2:
                    if kata_negatif_ditemukan:
                        kata_neg_str = ", ".join([f"`{k}`" for k in kata_negatif_ditemukan])
                        st.error(f"**Kata Negatif Terdeteksi ({jml_neg}):**\n\n{kata_neg_str}")
                    else:
                        st.info("**Kata Negatif Terdeteksi:**\n\n0 kata.")
                        
                if prediksi == "POSITIF" and (jml_neg > jml_pos):
                    st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Positif**, meskipun terdapat lebih banyak kata bernada negatif. Hal ini terjadi karena model IndoBERT memahami konteks kalimat utuh (misal: kata penyangkalan 'tidak buruk').")
                    
                elif prediksi == "NEGATIF" and (jml_pos > jml_neg):
                    st.warning("💡 **Catatan Analisis:** Model menyimpulkan ulasan ini **Negatif**, meskipun terdapat lebih banyak kata bernada positif. Harap perhatikan konteks kalimat (misal: sarkasme).")
                
                if user_input != teks_terjemahan:
                    st.markdown(f"<small style='color:gray;'>*Sistem mendeteksi bahasa asing. Kalimat diterjemahkan menjadi: '{teks_terjemahan}' sebelum diprediksi oleh AI.*</small>", unsafe_allow_html=True)

# FOOTER: INSIGHTS & DATA MENTAH
st.markdown("---")
st.subheader("💡 Insights & Rekomendasi Manajerial")
if 'sentiment' in df_working.columns:
    insights = generate_insights(df_working)
    for i, insight in enumerate(insights, 1):
        st.info(f"**{i}.** {insight}")

with st.expander("Lihat Data Ulasan Mentah (Tabel)"):
    # PERBAIKAN: Kolom 'sentiment' DIHAPUS dari tampilan tabel website, 
    # TAPI TETAP ADA di sistem (df_working) agar fungsi lain tidak error.
    cols_to_show = ['pelabuhan', 'tanggal', 'review_text', 'review_rating']
    available_cols = [c for c in cols_to_show if c in df_working.columns]
    
    if 'aspects' in df_working.columns:
        available_cols.append('aspects')
        
    st.dataframe(df_working[available_cols])
