"""Generates small, real-world demo datasets for the Streamlit UI."""

import logging
from pathlib import Path
from typing import cast

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_demo_data() -> None:
    """Download demo data (lite versions for ease)."""
    out_dir = Path("demo_data")
    out_dir.mkdir(exist_ok=True)

    try:
        # 1. Titanic Passenger Data (Classic analytical dataset, permanent URL)
        logger.info("Downloading Titanic dataset sample...")
        titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        titanic_df = pd.read_csv(titanic_url)

        # Select relevant columns and take a 100-row sample
        titanic_sample = titanic_df[
            ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Fare", "Embarked"]
        ].head(100)
        titanic_sample.to_csv(out_dir / "titanic_lite.csv", index=False)
        logger.info("✅ Saved demo_data/titanic_lite.csv (100 real rows)")

        # 2. Spotify Tracks (Real data via HuggingFace)
        logger.info("Downloading Spotify dataset sample...")
        from datasets import load_dataset  # type: ignore

        # Use the highly reliable datasets library
        spotify_ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        spotify_df = cast(pd.DataFrame, spotify_ds.to_pandas())

        # Select and rename columns, take a 100-row sample
        spotify_sample = spotify_df.head(100)[
            ["track_name", "artists", "track_genre", "danceability", "energy", "valence", "popularity"]
        ].rename(columns={"artists": "artist", "track_genre": "genre", "popularity": "popularity_score"})
        spotify_sample.to_csv(out_dir / "spotify_lite.csv", index=False)
        logger.info("✅ Saved demo_data/spotify_lite.csv (100 real rows)")

        logger.info("🎉 Demo datasets generated successfully with real, transparent data!")

    except Exception as e:
        logger.error(f"Failed to download demo data: {e}")
        logger.info("Please ensure you have an internet connection and the 'datasets' library installed.")


if __name__ == "__main__":
    generate_demo_data()
