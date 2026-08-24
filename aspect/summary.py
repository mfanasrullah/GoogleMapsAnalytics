import pandas as pd

class AspectSummarizer:
    @staticmethod
    def get_aspect_sentiment_summary(df, aspect_col='aspects', sentiment_col='sentiment', location_col='location'):
        """
        Mengubah DataFrame yang memiliki list aspek menjadi format yang 
        bisa divisualisasikan dengan mudah di Dashboard.
        """
      
        df_exploded = df.explode(aspect_col)
        
        summary = df_exploded.groupby([location_col, aspect_col, sentiment_col]).size().reset_index(name='count')
        
        return summary

    @staticmethod
    def get_top_pain_points(df, location_col='location'):
        """
        Mengembalikan aspek negatif tertinggi (Pain Points) untuk masing-masing pelabuhan.
        Asumsi dataframe sudah melewati proses explode.
        """
        if type(df['aspects'].iloc[0]) == list:
            df = df.explode('aspects')
            
        neg_df = df[df['sentiment'] == 'NEGATIVE']
        pain_points = neg_df.groupby([location_col, 'aspects']).size().reset_index(name='complaint_count')
        
        pain_points = pain_points.sort_values(['location', 'complaint_count'], ascending=[True, False])
        return pain_points
