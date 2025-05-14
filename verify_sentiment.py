# verify_sentiment.py
import pandas as pd
from src.data_loader   import load_reviews
from src.nlp_features  import nltk_sentiment

# 1) Chargez et nettoyez
df = load_reviews()

# 2) Appliquez VADER
df = nltk_sentiment(df)

# 3) Affichez un aperçu
pd.set_option('display.max_columns', None)
print("=== Aperçu des 10 premiers avis avec leur score VADER ===\n")
print(df[['text','sentiment']].head(10))

# 4) Vérifiez la répartition des scores
print("\n=== Distribution des sentiments VADER ===")
print(df['sentiment'].value_counts().sort_index())

# 5) Exemples d’avis jugés 1★ et 5★ par VADER
print("\n=== Exemples d’avis 1 étoile selon VADER ===")
print(df[df['sentiment']==1][['star_rating','text']].sample(3, random_state=0).to_string(index=False))

print("\n=== Exemples d’avis 5 étoiles selon VADER ===")
print(df[df['sentiment']==5][['star_rating','text']].sample(3, random_state=0).to_string(index=False))
