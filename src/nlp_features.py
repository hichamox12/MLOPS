import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Télécharger le lexique VADER si nécessaire
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def tfidf_features(df_reviews: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """
    Génère des features TF-IDF à partir du texte des avis.
    Retourne un DataFrame indexé comme df_reviews avec les colonnes TF-IDF + review_id, order_id.
    """
    vect = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vect.fit_transform(df_reviews["text"].fillna(""))
    df_tfidf = pd.DataFrame(
        X.toarray(),
        columns=vect.get_feature_names_out(),
        index=df_reviews.index
    )
    df_tfidf["review_id"] = df_reviews["review_id"]
    df_tfidf["order_id"]  = df_reviews["order_id"]
    return df_tfidf

def nltk_sentiment(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'sentiment' (1–5 étoiles) en utilisant VADER de NLTK.
    Mappe le score 'compound' [-1,1] sur l’échelle 1–5.
    """
    df = df_reviews.copy()
    scores = []
    for txt in df["text"].fillna(""):
        vs = sia.polarity_scores(txt)
        # mapping compound [-1,1] → [1,5]
        stars = round((vs["compound"] + 1) * 2 + 1)
        stars = min(max(stars, 1), 5)
        scores.append(stars)
    df["sentiment"] = scores
    return df
