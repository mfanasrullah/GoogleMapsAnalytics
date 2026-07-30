import pandas as pd

def generate_insights(df_processed):
    insights = []
    
    loc_col = 'pelabuhan' if 'pelabuhan' in df_processed.columns else 'location'
    
    neg_stats = df_processed[df_processed['sentiment'] == 'NEGATIVE'].groupby(loc_col).size()
    if not neg_stats.empty:
        worst_terminal = neg_stats.idxmax()
        insights.append(f"Fokus perbaikan utama direkomendasikan pada **{worst_terminal}** karena memiliki volume keluhan (sentimen negatif) tertinggi.")

    if 'aspects' in df_processed.columns:
        df_neg = df_processed[df_processed['sentiment'] == 'NEGATIVE'].copy()
        if not df_neg.empty:
            # Karena aspects berbentuk list, kita pecah (explode) terlebih dahulu
            df_neg_explode = df_neg.explode('aspects')
            neg_aspects = df_neg_explode['aspects'].value_counts()
            
            # Abaikan kategori 'Umum/Lainnya' agar rekomendasi lebih berbobot
            neg_aspects = neg_aspects[neg_aspects.index != 'Umum/Lainnya']
            
            if not neg_aspects.empty:
                top_issues = ", ".join(neg_aspects.head(2).index.tolist())
                insights.append(f"Isu operasional yang paling sering dikeluhkan oleh penumpang mencakup aspek: **{top_issues}**.")

    pos_stats = df_processed[df_processed['sentiment'] == 'POSITIVE'].groupby(loc_col).size()
    if not pos_stats.empty:
        best_terminal = pos_stats.idxmax()
        insights.append(f"**{best_terminal}** menunjukkan performa layanan terbaik (pujian terbanyak) dan dapat dijadikan *benchmark* pelayanan untuk terminal lainnya.")

    if not insights:
        insights.append("Belum ada data ulasan yang cukup untuk menghasilkan rekomendasi.")

    return insights
