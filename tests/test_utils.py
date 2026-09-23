
import pandas as pd
import pytest
from pathlib import Path
import shutil
import os

from src.hiatus.utils import (
    extract_score,
    df_to_jsonl,
    load_data,
    setup_dirs,
)

def test_extract_score():
    result = [{"label": "A", "score": 0.9}, {"label": "B", "score": 0.1}]
    assert extract_score(result, "A") == 0.9
    assert extract_score(result, "B") == 0.1
    with pytest.raises(ValueError):
        extract_score(result, "C")

def test_df_to_jsonl_and_load_data(tmp_path):
    df = pd.DataFrame([{"col1": "a", "col2": 1}, {"col1": "b", "col2": 2}])
    file_path = tmp_path / "test.jsonl"
    df_to_jsonl(df, str(file_path))
    
    loaded_df = load_data(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)

def test_setup_dirs(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dirs_to_create = {"dir1": dir1, "dir2": dir2}
    
    setup_dirs(dirs_to_create)
    
    assert dir1.is_dir()
    assert dir2.is_dir()

def test_df_to_jsonl_and_load_data_with_non_ascii(tmp_path):
    df = pd.DataFrame([{"text": "你好世界"}])
    file_path = tmp_path / "test_non_ascii.jsonl"
    df_to_jsonl(df, str(file_path))

    loaded_df = load_data(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)
    assert loaded_df["text"][0] == "你好世界"
