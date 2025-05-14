# train_trend.py
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.data_loader  import load_reviews
from src.nlp_features import nltk_sentiment
from src.features     import build_trend_dataset, add_time_lags

def main():
    # 1) Charger et enrichir les avis
    df_reviews = nltk_sentiment(load_reviews())

    # 2) Construire le dataset "trending" et obtenir le seuil
    df_trend, seuil = build_trend_dataset(df_reviews, quantile=0.5)

    # 3) Ajouter les variables de lags
    df_feat = add_time_lags(df_trend, n_lags=3)

    # 4) Préparer X et y pour la classification
    X = df_feat.drop(["product_id","month","trending"], axis=1)
    y = df_feat["trending"]

    # Séparation chronologique 80/20
    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 5) Entraîner le modèle
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=0
    )
    model.fit(X_train, y_train)

    # 6) Évaluer
    pred_test  = model.predict(X_test)
    proba_test = model.predict_proba(X_test)[:, 1]
    print("=== Classification report ===")
    print(classification_report(y_test, pred_test))
    print("ROC AUC :", roc_auc_score(y_test, proba_test))
    print(f"Seuil trending (quantile=0.5) : {seuil:.0f}")

    # 7) Sauvegarder le modèle et le seuil
    joblib.dump((model, seuil), "model_xgb_trend.pkl")
    print("Modèle enregistré sous → model_xgb_trend.pkl")

if __name__ == "__main__":
    main()
