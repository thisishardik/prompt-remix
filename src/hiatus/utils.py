from typing import Dict
from pathlib import Path

import pandas as pd

def extract_score(result, label):
    for res in result:
        if res["label"] == label:
            return res["score"]
    raise ValueError(f"Label {label} not found")

def df_to_jsonl(df: pd.DataFrame, fname: str) -> None:
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, index=False, lines=True, orient="records", force_ascii=False)

def load_data(fname: str) -> pd.DataFrame:
    with open(fname, "r") as file:
        df = pd.read_json(file, lines=True)
    return df

def setup_dirs(dict: Dict[str, Path]) -> None:
    for path in dict.values():
        path.mkdir(parents=True, exist_ok=True)        
