import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_databench_lite(target_count=5):
    out_dir = Path("data/databench_lite")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading DataBench SemEval split to get dataset IDs...")
    hf_dataset = load_dataset("cardiffnlp/databench", "semeval", split="train")
    
    # Gather unique dataset IDs, preserving order of first appearance
    dataset_ids = list(dict.fromkeys(row["dataset"] for row in hf_dataset if row.get("dataset"))) # type: ignore
    
    saved = 0
    for dataset_id in dataset_ids:
        try:
            parquet_url = f"https://huggingface.co/datasets/cardiffnlp/databench/resolve/main/data/{dataset_id}/sample.parquet"
            df = pd.read_parquet(parquet_url)
            csv_path = out_dir / f"{dataset_id}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved DataBench Lite dataset: %s.csv (Rows: %d)", dataset_id, len(df))
            saved += 1
            if saved >= target_count:
                logger.info("Reached target of %d datasets. Stopping.", target_count)
                break
        except Exception as e:
            logger.warning("Skipping %s: %s", dataset_id, e)
            # Do not increment saved; just move to the next unique dataset

if __name__ == "__main__":
    download_databench_lite()