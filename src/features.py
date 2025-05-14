import pandas as pd

__all__ = [
    "aggregate_reviews_monthly",
    "build_trend_dataset",
    "add_time_lags"
]

def aggregate_reviews_monthly(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les avis par produit et par mois:
      - nb_reviews    = nombre d’avis
      - avg_rating    = moyenne des star_rating
      - avg_sentiment = moyenne des sentiment
    """
    df = df_reviews.copy()
    df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()
    return (
        df.groupby(["product_id","month"]).agg(
            nb_reviews    = ("sentiment","size"),
            avg_rating    = ("star_rating","mean"),
            avg_sentiment = ("sentiment","mean"),
        ).reset_index()
    )


def build_trend_dataset(df_reviews: pd.DataFrame, quantile: float = 0.5) -> (pd.DataFrame, float):
    """
    Construit un dataset pour classification binaire "trending":
      - product_id, month, nb_reviews, avg_rating, avg_sentiment
      - trending = 1 si nb_reviews le mois suivant > seuil (quantile)
    Renvoie le DataFrame et le seuil utilisé.
    """
    rev_m = aggregate_reviews_monthly(df_reviews)
    rev_m["nb_reviews_next"] = (
        rev_m.groupby("product_id")["nb_reviews"].shift(-1)
    )
    seuil = rev_m["nb_reviews_next"].quantile(quantile)
    rev_m = rev_m.dropna(subset=["nb_reviews_next"]).reset_index(drop=True)
    rev_m["trending"] = (rev_m["nb_reviews_next"] > seuil).astype(int)
    return rev_m.drop(columns=["nb_reviews_next"]), seuil


def add_time_lags(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """
    Ajoute des lags du nombre d’avis et une moyenne mobile sur 3 mois.
    """
    df_out = df.copy()
    for lag in range(1, n_lags+1):
        df_out[f"nb_reviews_lag_{lag}"] = (
            df_out.groupby("product_id")["nb_reviews"].shift(lag)
        )
    df_out["nb_rev_ma_3"] = (
        df_out.groupby("product_id")["nb_reviews"]
              .rolling(3).mean()
              .reset_index(level=0, drop=True)
    )
    return df_out.dropna().reset_index(drop=True)