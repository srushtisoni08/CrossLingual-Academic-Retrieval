from pathlib import Path
from datasets import load_dataset

# Define project paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Target languages
LANGUAGES = ["en", "bn", "te", "hi"]
DATASET_NAME = "nlpai-lab/miracl-multilingual-triplets"

def download_and_save():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        print(f"Downloading {lang} dataset...")
        # Load the dataset split from Hugging Face
        ds = load_dataset(DATASET_NAME, lang)

        # Path to save language dataset
        lang_dir = RAW_DATA_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Save train split to CSV and Parquet formats
        for split in ds.keys():
            df = ds[split].to_pandas()

            # Save as CSV
            csv_path = lang_dir / f"{split}.csv"
            df.to_csv(csv_path, index=False)

            # Save as Parquet (optimized for fast reading in pandas/polars)
            parquet_path = lang_dir / f"{split}.parquet"
            df.to_parquet(parquet_path, index=False)

            print(f" Saved {lang} ({split}) to {lang_dir}")

if __name__ == "__main__":
    download_and_save()