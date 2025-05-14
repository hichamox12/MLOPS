import pandas as pd
import csv
from functools import lru_cache

@lru_cache(maxsize=1)
def load_reviews(path: str = "data/customer_reviews.csv") -> pd.DataFrame:
    """
    Charge et retourne le DataFrame complet des avis clients nettoyés.
    Opérations réalisées :
      - Lecture robuste du CSV (gestion des lignes malformées).
      - Conversion de 'review_date' en datetime et suppression des invalides.
      - Renommage de la colonne 'review_body' en 'text'.
      - Suppression des doublons exacts.
      - Suppression des espaces superflus dans toutes les colonnes de type texte.
      - Conversion de 'star_rating' en numérique.
    """
    # 1) Lecture sans parse_dates
    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"'
    )

    # 2) Forcer conversion de review_date en datetime
    df["review_date"] = pd.to_datetime(
        df.get("review_date"),
        dayfirst=False,    # vos dates sont au format YYYY-MM-DD
        errors="coerce"    # convertit en NaT les valeurs invalides
    )
    # 3) Supprimer les lignes sans date valide
    df = df.dropna(subset=["review_date"]).reset_index(drop=True)

    # 4) Renommer review_body → text
    if "review_body" in df.columns:
        df.rename(columns={"review_body": "text"}, inplace=True)

    # 5) Suppression de doublons et nettoyage d’espaces
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].str.strip()

    # 6) Conversion star_rating → float
    if "star_rating" in df.columns:
        df["star_rating"] = (
            df["star_rating"].astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float)
        )

    return df
