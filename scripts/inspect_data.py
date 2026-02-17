"""Inspect raw datasets to understand their structure."""

import pandas as pd
from pathlib import Path

data_dir = Path("data/raw")

files = [
    "Instagram_Analytics.csv",
    "Social Media Engagement Dataset.csv",
    "social_media_viral_content_dataset.csv",
]

for filename in files:
    filepath = data_dir / filename
    if filepath.exists():
        print(f"\n{'='*60}")
        print(f"FILE: {filename}")
        print('='*60)

        df = pd.read_csv(filepath)
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nFirst 2 rows:\n{df.head(2)}")
    else:
        print(f"\nFile not found: {filename}")
