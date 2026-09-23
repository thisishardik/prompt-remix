
import pytest
from pathlib import Path

from src.hiatus.data import hrs_paths, translated_paths

@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setattr("hiatus.data.common_storage_dir", "/fake/storage")
    monkeypatch.setattr("hiatus.data.privatized_dir", "/fake/privatized")
    monkeypatch.setattr("hiatus.data.data_dir", "/fake/data")

def test_hrs_paths_phase2_en_cross_genre(mock_config):
    paths = hrs_paths(genre="en_crossGenre", language="en")
    assert paths["TA1"]["groundtruth"] == Path("/fake/storage/hrs2/mode_crossGenre/TA1/hrs_06-27-24_english_crossGenre-combined/groundtruth/hrs_06-27-24_english_crossGenre-combined_TA1_groundtruth.jsonl")

def test_hrs_paths_phase3_zh_short_cross_genre(mock_config):
    paths = hrs_paths(genre="zh_short_crossGenre", language="zh")
    assert paths["TA3"]["data"]["privatized"] == Path("/fake/privatized/zh/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA3_privatized_queries_prompt-remix.jsonl")

def test_hrs_paths_phase3_ar(mock_config):
    paths = hrs_paths(genre="AMINA", language="ar")
    assert paths["TA1"]["data"] == Path("/fake/storage/nonhrs/AMINA/TA1/AMINA/data")

def test_hrs_paths_invalid_combination(mock_config):
    with pytest.raises(ValueError):
        hrs_paths(genre="en_crossGenre", language="zh")

def test_translated_paths(mock_config):
    input_path = Path("/fake/data/hrs2/queries/en/some_file.jsonl")
    output_path = translated_paths(input_path, source="en", target="ar")
    assert output_path == Path("/fake/data/hrs2/queries/ar/translated/en/some_file.jsonl")

def test_hrs_paths_per_genre(mock_config):
    paths = hrs_paths(genre="perGenre-HRS2.1", language="en")
    assert paths["TA1"]["groundtruth"] == Path("/fake/storage/hrs2/mode_perGenre-HRS2.1/TA1/hrs_06-27-24_english_perGenre-HRS2.1/groundtruth/hrs_06-27-24_english_perGenre-HRS2.1_TA1_groundtruth.jsonl")
