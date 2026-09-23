
import pytest
import dataclasses
import json
import numpy as np
import pickle
import gzip
import bz2
from pathlib import Path

from src.hiatus.evaluation.io_utils import (
    ComplexEncoder,
    JsonObject,
    read_file_to_set,
    read_file_to_list,
    write_list_to_file,
    serialize_to_npz,
    deserialize_from_npz,
    dict2string,
    serialize_to_pickle,
    deserialize_from_pickle,
    fopen,
    JSONLineVisitor,
    JSONLWithCkpts,
)

@dataclasses.dataclass
class MyDataClass:
    x: int
    y: str

class MyJsonObject(JsonObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b

def test_complex_encoder():
    assert json.dumps(MyDataClass(x=1, y="a"), cls=ComplexEncoder) == '{"x": 1, "y": "a"}'
    assert json.dumps(MyJsonObject(a=1, b="test"), cls=ComplexEncoder) == '{"a": 1, "b": "test"}'

def test_json_object():
    obj = MyJsonObject(a=1, b="test")
    assert obj.reprJSON() == {"a": 1, "b": "test"}

def test_read_write_list_set(tmp_path):
    lines = ["a", "b", "c"]
    file = tmp_path / "test.txt"
    write_list_to_file(lines, file)
    
    read_lines_list = read_file_to_list(file)
    assert read_lines_list == lines

    read_lines_set = read_file_to_set(file)
    assert read_lines_set == set(lines)

def test_serialize_deserialize_npz(tmp_path):
    data = np.array([1, 2, 3])
    file = tmp_path / "test.npz"
    serialize_to_npz(data, file)
    
    loaded_data = deserialize_from_npz(file)
    np.testing.assert_array_equal(data, loaded_data)

def test_dict2string():
    d = {"a": 1, "b": 2}
    expected = "\n\tMy Dict\n\t\ta=1\n\t\tb=2\n"
    assert dict2string(d, "My Dict") == expected

def test_serialize_deserialize_pickle(tmp_path):
    data = {"a": 1, "b": "hello"}
    file = tmp_path / "test.pkl.gz"
    serialize_to_pickle(data, file)
    
    loaded_data = deserialize_from_pickle(file)
    assert data == loaded_data

def test_fopen(tmp_path):
    # Test with regular file
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("hello")
    with fopen(str(file)) as f:
        assert f.read() == "hello"

    # Test with .gz file
    gz_file = tmp_path / "test.txt.gz"
    with gzip.open(gz_file, "wt") as f:
        f.write("hello_gz")
    with fopen(str(gz_file)) as f:
        assert f.read() == "hello_gz"

    # Test with .bz2 file
    bz2_file = tmp_path / "test.txt.bz2"
    with bz2.open(bz2_file, "wt") as f:
        f.write("hello_bz2")
    with fopen(str(bz2_file)) as f:
        assert f.read() == "hello_bz2"

def test_json_line_visitor(tmp_path):
    file = tmp_path / "test.jsonl"
    lines = [{"a": 1}, {"b": 2}]
    with open(file, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    visitor = JSONLineVisitor(str(file))
    visitor.init_mmap_handle()
    visitor.build_index()
    assert len(visitor) == 2
    assert visitor[0] == {"a": 1}
    assert visitor[1] == {"b": 2}
    all_lines = [line for line in visitor]
    assert all_lines == lines
    visitor.release_mmap_handle()

def test_jsonl_with_ckpts(tmp_path):
    file = tmp_path / "test.jsonl"
    
    with JSONLWithCkpts(str(file)) as writer:
        writer.write_line("id1", {"data": "a"})
        assert not writer.should_skip_line("id2")
        assert writer.should_skip_line("id1")

    with open(file, "r") as f:
        assert json.loads(f.read()) == {"line_id": "id1", "data": {"data": "a"}}

    with JSONLWithCkpts(str(file)) as writer:
        assert writer.should_skip_line("id1")
        writer.write_line("id2", {"data": "b"})

    with open(file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"line_id": "id1", "data": {"data": "a"}}
        assert json.loads(lines[1]) == {"line_id": "id2", "data": {"data": "b"}}

