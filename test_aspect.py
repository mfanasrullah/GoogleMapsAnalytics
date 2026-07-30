import pandas as pd
from aspect.aspect import AspectExtractor
from aspect.summary import AspectSummarizer

df_reviews = pd.DataFrame({
    'location': ['Batam Centre Ferry Terminal', 'Harbour Bay Ferry Terminal'],
    'clean_text': ['tiket mahal dan antrean imigrasi lama', 'parkiran luas dan ruang tunggu nyaman'],
    'sentiment': ['NEGATIVE', 'POSITIVE']
})

extractor = AspectExtractor(method='rule-based') # Ubah ke 'zero-shot' jika ingin akurasi semantik
df_with_aspects = extractor.process_dataframe(df_reviews, text_column='clean_text')

summarizer = AspectSummarizer()
df_summary = summarizer.get_aspect_sentiment_summary(df_with_aspects)
df_pain_points = summarizer.get_top_pain_points(df_with_aspects)

print("Data dengan Aspek:")
print(df_with_aspects[['clean_text', 'aspects']])
print("\nRingkasan Aspek Sentimen:")
print(df_summary)
