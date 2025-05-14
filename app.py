import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import matplotlib.pyplot as plt

from src.data_loader import load_reviews
from src.nlp_features import nltk_sentiment
from src.features import build_trend_dataset, add_time_lags
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve

# Préparation NLTK
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Chatbot & Dashboard Tendances", layout="wide")
st.title("🤖 Chatbot & Dashboard Produits Tendances")

@st.cache_data
def load_data():
    # Chargement et sentiment
    df_raw = load_reviews()
    df = nltk_sentiment(df_raw)
    # Construction du dataset trending
    df_trend, threshold = build_trend_dataset(df, quantile=0.5)
    # Ajout des lags et features temporelles
    df_feat = add_time_lags(df_trend, n_lags=3)
    # Croissance relative 1 mois
    df_feat["rev_growth_1"] = (
        (df_feat["nb_reviews"] - df_feat["nb_reviews_lag_1"]) /
        df_feat["nb_reviews_lag_1"].replace(0, np.nan)
    ).fillna(0)
    return df, df_trend, df_feat, threshold

# Chargement des données
df, df_trend, df_feat, threshold = load_data()

# Produits/mois avec au moins un avis négatif
neg_df = df[df["sentiment"] <= 3]
neg_set = set(zip(
    neg_df["product_id"],
    neg_df["review_date"].dt.to_period("M").dt.to_timestamp()
))

# --- Chatbot Section ---
st.markdown("---")
st.header("🤖 Chatbot Produits & Tendances")
query = st.text_area("Posez votre question…", height=100)

def answer(q: str) -> str:
    # Ne pas tout mettre en minuscules pour la capture
    q_low = q.lower()
    q_orig = q  # on garde la casse d’origine pour l’extraction

    # 1) Extraction stricte de l’ASIN : 10 caractères alphanumériques (MAJ + chiffres)
    m = re.search(r"product_id\s*=\s*([A-Za-z0-9]{10})", q_orig)
    if m and "tendance" in q_low:
        pid = m.group(1).upper()  # force la majuscule

        # 2) Vérifier l’existence côté df (on passe tout en upper pour comparer)
        prod_ids = df["product_id"].str.upper()
        if pid not in prod_ids.values:
            return f"💬 Le product_id={pid} n'existe pas dans notre base de données."

        # 3) Récupérer la dernière ligne du mois
        last = df_feat["month"].max()
        row = df_feat[
            (df_feat["product_id"].str.upper() == pid) &
            (df_feat["month"] == last)
        ]
        if row.empty:
            return f"💬 Pas de données pour product_id={pid} au mois {last.date()}."

        # 4) Appliquer ta règle rating ≥4 & sentiment ≥4
        avg_r = row["avg_rating"].iloc[0]
        avg_s = row["avg_sentiment"].iloc[0]
        if avg_r >= 4 and avg_s >= 4:
            return (f"💬 ✅ Le produit {pid} est en tendance "
                    f"(avg_rating={avg_r:.1f}, avg_sentiment={avg_s:.1f}).")
        else:
            return (f"💬 ❌ Le produit {pid} n'est pas en tendance "
                    f"(avg_rating={avg_r:.1f}, avg_sentiment={avg_s:.1f}).")

    # ... le reste de tes intents inchangé ...


    # 3) Avis positifs/négatifs pour un product_id
    if m and "avis positifs" in q_low:
        pid = m.group(1)
        # Vérification si le produit existe
        if pid not in df["product_id"].values:
            return f"💬 Le product_id={pid} n'existe pas dans notre base de données."
        revs = df[df["product_id"] == pid]
        total = len(revs)
        pos = len(revs[revs["sentiment"] >= 4])
        return f"💬 Produit {pid} : {pos}/{total} avis positifs."
    
    if m and ("avis négatifs" in q_low or "avis negatifs" in q_low):
        pid = m.group(1)
        # Vérification si le produit existe
        if pid not in df["product_id"].values:
            return f"💬 Le product_id={pid} n'existe pas dans notre base de données."
        revs = df[df["product_id"] == pid]
        neg = len(revs[revs["sentiment"] <= 3])
        return f"💬 Produit {pid} : {neg}/{len(revs)} avis négatifs."

    # 4) Nombre d'avis par mois
    MONTHS = {
        "janvier":1, "février":2, "fevrier":2, "mars":3,
        "avril":4, "mai":5, "juin":6, "juillet":7,
        "août":8, "aout":8, "septembre":9,
        "octobre":10, "novembre":11, "décembre":12, "decembre":12
    }
    for name, num in MONTHS.items():
        if name in q_low:
            cnt = df[df["review_date"].dt.month == num].shape[0]
            return f"💬 Il y a {cnt} avis en {name.capitalize()}."
    
    # 5) Nouvelle fonctionnalité: Informations sur un produit
    if m and "info" in q_low:
        pid = m.group(1)
        if pid not in df["product_id"].values:
            return f"💬 Le product_id={pid} n'existe pas dans notre base de données."
        
        prod_data = df[df["product_id"] == pid]
        n_reviews = len(prod_data)
        avg_rating = prod_data["star_rating"].mean()
        avg_sentiment = prod_data["sentiment"].mean()
        first_date = prod_data["review_date"].min().date()
        last_date = prod_data["review_date"].max().date()
        
        return (f"💬 Produit {pid}:\n"
                f"- Nombre d'avis: {n_reviews}\n"
                f"- Note moyenne: {avg_rating:.2f}/5\n"
                f"- Sentiment moyen: {avg_sentiment:.2f}/5\n"
                f"- Premier avis: {first_date}\n"
                f"- Dernier avis: {last_date}")
    
    return "💬 Désolé, je n'ai pas compris la question."

if st.button("Envoyer"):
    st.markdown(answer(query))

# --- Débogage produit ---
#with st.expander("🔍 Débogage produit"):
 #   product_id_debug = st.text_input("ID du produit à déboguer", "b000bqym0w")
    
  #  if st.button("Déboguer"):
        # Vérification dans DataFrame original
   #     in_df = product_id_debug in df["product_id"].values
    #    st.write(f"1) Présent dans DataFrame original (df): {in_df}")
     #   if in_df:
      #      prod_df = df[df["product_id"] == product_id_debug]
       #     st.write(f"   Nombre d'avis: {len(prod_df)}")
        #    st.write(f"   Mois disponibles: {sorted(prod_df['review_date'].dt.to_period('M').unique())}")
        #    st.write(f"   Note moyenne: {prod_df['star_rating'].mean():.2f}")
        #    st.write(f"   Sentiment moyen: {prod_df['sentiment'].mean():.2f}")
        
        # Vérification dans DataFrame de tendances
       # in_df_trend = product_id_debug in df_trend["product_id"].values
       # st.write(f"2) Présent dans DataFrame tendance (df_trend): {in_df_trend}")
       # if in_df_trend:
        #    st.write(f"   Lignes disponibles: {len(df_trend[df_trend['product_id'] == product_id_debug])}")
        
        # Vérification dans DataFrame final avec lags
       # in_df_feat = product_id_debug in df_feat["product_id"].values
       # st.write(f"3) Présent dans DataFrame final (df_feat): {in_df_feat}")
       # if in_df_feat:
        #    st.write(f"   Lignes disponibles: {len(df_feat[df_feat['product_id'] == product_id_debug])}")
          #  st.write(f"   Mois disponibles: {sorted(df_feat[df_feat['product_id'] == product_id_debug]['month'].dt.date)}")
        
        # Informations supplémentaires pour le débogage
       # st.write(f"Seuil d'avis mensuel (threshold): {threshold:.0f}")
       # st.write(f"Dernier mois disponible dans df_feat: {df_feat['month'].max().date()}")

# --- Dashboard Section ---
st.markdown("---")
st.header("📊 KPI & Visualisations")
# 1) KPI des avis
col1, col2, col3 = st.columns(3)
col1.metric("Nombre d'avis", len(df))
col2.metric("Note moyenne clients", f"{df['star_rating'].mean():.2f} ⭐")
col3.metric("Sentiment moyen VADER", f"{df['sentiment'].mean():.2f} ⭐")

st.markdown("---")
# 2) Répartition des notes et sentiments
st.subheader("Répartition des notes clients (1–5)")
st.bar_chart(df['star_rating'].value_counts().sort_index())
st.subheader("Répartition du sentiment VADER (1–5)")
st.bar_chart(df['sentiment'].value_counts().sort_index())

# 3) Histogramme des longueurs des avis
#st.subheader("Distribution de la longueur des avis")
#lengths = df['text'].str.len()
#fig, ax = plt.subplots()
#ax.hist(lengths, bins=20)
#ax.set_xlabel("Longueur (caractères)")
#ax.set_ylabel("Nombre d'avis")
#st.pyplot(fig)

# 4) Aperçu interactif des avis
st.markdown("---")
st.sidebar.subheader("Paramètres d'affichage")
n_rows = st.sidebar.slider("Nombre de lignes à afficher", 10, len(df), 40, 10)
st.subheader(f"Aperçu des {n_rows} premières lignes des avis")
st.dataframe(df.head(n_rows), use_container_width=True)

# 5) Prédiction des produits tendances
st.markdown("---")
st.header("🔮 Prédiction des produits tendances")
try:
    model, seuil = joblib.load("model_xgb_trend.pkl")
except FileNotFoundError:
    st.warning("⚠️ Modèle non trouvé : exécutez `python train_trend.py`. ")
else:
    df_trend2, _ = build_trend_dataset(df, quantile=0.5)
    df_feat2 = add_time_lags(df_trend2, n_lags=3)
    last_month = df_feat2['month'].max()
    df_last = df_feat2[df_feat2['month'] == last_month].copy()
    cols_to_drop = [c for c in ["product_id","month","trending"] if c in df_last.columns]
    X_last = df_last.drop(cols_to_drop, axis=1)
    df_last['forecast_trending'] = model.predict(X_last)

    st.subheader("Produits prévus en tendance")
    st.write(f"Seuil d'avis mensuel = {seuil:.0f} avis")
    st.dataframe(
        df_last[df_last['forecast_trending'] == 1][
            ["product_id","nb_reviews","avg_rating","avg_sentiment"]
        ], use_container_width=True
    )

    # Visualisation performance
    df_all, _ = build_trend_dataset(df, quantile=0.5)
    df_all_feat = add_time_lags(df_all, n_lags=3)
    split = int(len(df_all_feat) * 0.8)
    X = df_all_feat.drop(["product_id","month","trending"] if True else [], axis=1)
    y = df_all_feat['trending']
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_score = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"ROC AUC = {roc_score:.2f}")
    ax2.plot([0,1],[0,1], linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve du classifieur")
    ax2.legend(loc='lower right')
    st.pyplot(fig2)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig3, ax3 = plt.subplots()
    disp.plot(ax=ax3)
    ax3.set_title("Matrice de confusion")
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("🔍 Probabilité de tendance vs Sentiment moyen")
    df_last['prob_trend'] = model.predict_proba(X_last)[:,1]
    fig5, ax5 = plt.subplots()
    scatter = ax5.scatter(
        df_last['avg_sentiment'], df_last['prob_trend'],
        c=df_last['forecast_trending'], cmap='coolwarm', edgecolor='k', alpha=0.7
    )
    ax5.set_xlabel('Sentiment moyen (1–5)')
    ax5.set_ylabel('Probabilité prédite de tendance')
    ax5.set_title('Probabilité de tendance vs Sentiment moyen')
    legend1 = ax5.legend(*scatter.legend_elements(), title='Tendance')
    ax5.add_artist(legend1)
    st.pyplot(fig5)

    st.markdown("---")
    st.subheader("📈 Courbe d'apprentissage du modèle")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1,1.0,5)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig6, ax6 = plt.subplots()
    ax6.plot(train_sizes, train_mean, 'o-', label='Train accuracy')
    ax6.plot(train_sizes, test_mean, 'o-', label='CV accuracy')
    ax6.set_xlabel('Taille de échantillon d entraînement')
    ax6.set_ylabel('Accuracy')
    ax6.set_title("Courbe d'apprentissage")
    ax6.legend(loc='best')
    st.pyplot(fig6)