import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def poison_titanic():
    input_path = Path("data/Titanic-Dataset.csv")
    out_dir = Path("data/poisoned")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Titanic dataset not found at %s", input_path)
        return

    df = pd.read_csv(input_path)

    # 1. String artifacts in a numeric column
    df_art = df.copy()
    
    # Explicitly cast to object so Pandas 3.x allows mixed types in the column
    df_art['Fare'] = df_art['Fare'].astype(object)
    
    # Add '$' to the first 10 fares, and 'N/A' to the next 10
    df_art.loc[0:10, 'Fare'] = "$" + df_art['Fare'][0:11].astype(str)
    df_art.loc[11:20, 'Fare'] = "N/A"
    
    out_art = out_dir / "titanic_artifacts.csv"
    df_art.to_csv(out_art, index=False)
    logger.info("Created artifact-poisoned dataset: %s", out_art)

    # 2. Semicolon delimited (despite .csv extension)
    out_semi = out_dir / "titanic_semicolon.csv"
    df.to_csv(out_semi, sep=";", index=False)
    logger.info("Created semicolon-delimited dataset: %s", out_semi)

    # 3. UTF-16 Encoding
    out_utf16 = out_dir / "titanic_utf16.csv"
    df.to_csv(out_utf16, encoding="utf-16", index=False)
    logger.info("Created UTF-16 encoded dataset: %s", out_utf16)

if __name__ == "__main__":
    poison_titanic()