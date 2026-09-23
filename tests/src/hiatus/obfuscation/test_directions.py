
import pandas as pd
import numpy as np
import pytest

from src.hiatus.obfuscation.directions import (
    std_to_weight,
    choose_directions_genre_mean,
    choose_sliders_round_robin,
    choose_directions_target_author,
    combine_directions,
)

def test_std_to_weight():
    assert std_to_weight(0.5) == 0.5
    assert std_to_weight(1.5) == 0.7
    assert std_to_weight(2.5) == 0.9
    assert std_to_weight(3.5) == 1.0

def test_choose_directions_genre_mean():
    row = pd.Series({"style1": 2.5, "style2": -0.5, "type_a": 1.5, "type_b": -3.0})
    directions = choose_directions_genre_mean(row, top_n=2, available_types=["type_a", "type_b"], all_types=["type_a", "type_b"])
    assert len(directions) == 2
    # The two axes with the largest absolute values are type_b (3.0) and style1 (2.5)
    # The sign is flipped for non-type axes
    assert "style1" in directions and directions["style1"] < 0
    # For type axes a random one is chosen
    assert "type_a" in directions or "type_b" in directions


def test_choose_sliders_round_robin():
    styles = pd.DataFrame({
        "style1": [1.0, 2.0, 3.0],
        "style2": [3.0, 2.0, 1.0],
    }, index=["author1", "author2", "author3"])
    standard_devs = pd.Series({"style1": 1.0, "style2": 1.0})
    sliders = choose_sliders_round_robin(styles, top_n=1, available_types=[], all_types=[], standard_devs=standard_devs, closest=True)
    assert len(sliders) == 3
    assert "author1" in sliders
    assert "author2" in sliders
    assert "author3" in sliders
    assert isinstance(sliders["author1"], dict)

def test_choose_directions_target_author():
    styles = pd.DataFrame({
        "style1": [1.0, 3.0],
        "style2": [3.0, 1.0],
        "type_a": [0.1, 0.9]
    }, index=["author1", "target_author"])
    standard_devs = pd.Series({"style1": 1.0, "style2": 1.0, "type_a": 0.5})
    
    sliders = choose_directions_target_author(
        styles, top_n=2, available_types=["type_a"], all_types=["type_a"], standard_devs=standard_devs, target_author_id="target_author"
    )
    
    assert "author1" in sliders
    assert len(sliders["author1"]) == 2

    # Test mix_toward=False
    sliders_away = choose_directions_target_author(
        styles, top_n=2, available_types=["type_a"], all_types=["type_a"], standard_devs=standard_devs, target_author_id="target_author", mix_toward=False
    )
    # The weights should have opposite signs
    for axis in sliders["author1"]:
        if axis in sliders_away["author1"]:
            assert np.sign(sliders["author1"][axis]) == -np.sign(sliders_away["author1"][axis])


def test_combine_directions():
    directions = [
        {"style1": 0.5, "style2": -0.7},
        {"style1": 0.3, "type_a": 0.9},
    ]
    combined = combine_directions(directions, all_types=["type_a"])
    assert "style1" in combined
    assert "style2" in combined
    assert "type_a" in combined
    assert combined["style1"] == pytest.approx(0.4)
    assert combined["style2"] == -0.7
    assert combined["type_a"] == 0.9
