import bz2
import codecs
import dataclasses
import gzip
import io
import json
import mmap
import os
import pickle
import sys

import numpy as np

from hiatus.evaluation import logger

# copied from  https://ami-gitlab-01.bbn.com/text-group/nlplingo/-/blob/master/nlplingo/common/io_utils.py


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "reprJSON"):
            return obj.reprJSON()
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class JsonObject:
    # Serialization helper
    def reprJSON(self):
        d = dict()
        for a, v in self.__dict__.items():
            if v is None:
                continue
            if hasattr(v, "reprJSON"):
                d[a] = v.reprJSON()
            else:
                d[a] = v
        return d


def read_file_to_set(filename):
    ret = set()
    with codecs.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            ret.add(line.strip())
    return ret


def read_file_to_list(filename):
    with codecs.open(filename, "r", encoding="utf-8") as f:
        ret = [line.strip() for line in f]
    return ret


def write_list_to_file(lines, filepath):
    with codecs.open(filepath, "w", encoding="utf-8") as o:
        for line in lines:
            o.write(line)
            o.write("\n")


def safelog(s, logger_object):
    logger_object.info(s.encode("ascii", "ignore"))


def serialize_to_npz(python_obj, output_path):
    logger.info("Saving {}".format(output_path))
    np.savez_compressed(output_path, data=python_obj)


def deserialize_from_npz(input_path):
    logger.info("Loading {}".format(input_path))
    with np.load(input_path, allow_pickle=True) as fp2:
        return fp2["data"]


def dict2string(dictionary: dict, title: str):
    """Turns a dictionary into something pretty to print or write.

    Args:
        dictionary (dict): Dictionary with the data.
        title (str): Pretty title to give the collection of data.
    """
    message = "\n\t" + title + "\n"
    info = [
        "\t\t" + str(key) + "=" + str(value) + "\n"
        for key, value in list(dictionary.items())
    ]
    info_str = "".join(info)
    message += info_str

    return message


def serialize_to_pickle(obj, output_path):
    with gzip.open(output_path, mode="wb") as wfp:
        pickle.dump(obj, wfp)


def deserialize_from_pickle(input_path):
    with gzip.open(input_path, mode="rb") as rfp:
        return pickle.load(rfp)


def fopen(filename, mode="rt", encoding="utf-8", **kwargs):
    """Drop-in replacement for built in open() so that .gz and .bz2 files can be
    handled transparently. If filename is '-', standard input will be used.

    Since we are mostly dealing with text files UTF-8 encoding is used by default.
    """

    if filename == "-":
        if "w" in mode:
            return io.TextIOWrapper(sys.stdout.buffer, encoding=encoding)
        else:
            return io.TextIOWrapper(sys.stdin.buffer, encoding=encoding)

    if filename.endswith(".gz"):
        _fopen = gzip.open
        if "b" not in mode and "t" not in mode:
            mode = mode + "t"  # 'rb' is the default for gzip and bz2
    elif filename.endswith(".bz2"):
        _fopen = bz2.open
        if "b" not in mode and "t" not in mode:
            mode = mode + "t"
    else:
        _fopen = open
    if "b" in mode:
        return _fopen(filename, mode=mode, **kwargs)
    else:
        return _fopen(filename, mode=mode, encoding=encoding, **kwargs)


# class JSONLineVisitor():
#     """
#     Assuming you have a huge jsonl file and you want random access to any line of that file
#     """
#
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.mmap_handle = None
#         self.line_key_to_line_info = list()  # key is line key and value is a tuple of (num_bytes_to_seek, num_bytes_to_read)
#
#     def init_mmap_handle(self):
#         with fopen(self.file_path, 'rb') as fp:
#             self.mmap_handle = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
#
#     def release_mmap_handle(self):
#         self.mmap_handle.close()
#         self.mmap_handle = None
#
#     def build_index(self):
#         num_bytes_to_seek = 0
#         self.line_key_to_line_info.clear()
#         self.mmap_handle.seek(0, os.SEEK_SET)
#         line = self.mmap_handle.readline()
#         line_idx = 0
#         while len(line) > 0:
#             num_bytes_to_read = len(line)
#             self.line_key_to_line_info.append((num_bytes_to_seek, num_bytes_to_read))
#             num_bytes_to_seek += num_bytes_to_read
#             line_idx += 1
#             line = self.mmap_handle.readline()
#         assert line_idx == len(self.line_key_to_line_info)
#
#     def access_line(self, line_no):
#         num_bytes_to_seek, num_bytes_to_read = self.line_key_to_line_info[line_no]
#         self.mmap_handle.seek(num_bytes_to_seek, os.SEEK_SET)
#         line = self.mmap_handle.read(num_bytes_to_read).decode("utf-8")
#         json_line = json.loads(line)
#         return json_line
#
#     def length(self):
#         return len(self.line_key_to_line_info)
#
#     def __len__(self):
#         return self.length()
#
#     def __getitem__(self, item):
#         return self.access_line(item)


class JSONLineVisitor:
    """
    Assuming you have a huge jsonl file and you want random access to any line of that file
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.mmap_handle = None
        self.line_key_to_line_info = (
            list()
        )  # key is line key and value is a tuple of (num_bytes_to_seek, num_bytes_to_read)

    def init_mmap_handle(self):
        with fopen(self.file_path, "rb") as fp:
            self.mmap_handle = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    def release_mmap_handle(self):
        self.mmap_handle.close()
        self.mmap_handle = None

    def build_index(self):
        num_bytes_to_seek = 0
        self.line_key_to_line_info.clear()
        self.mmap_handle.seek(0, os.SEEK_SET)
        line = self.mmap_handle.readline()
        line_idx = 0
        while len(line) > 0:
            num_bytes_to_read = len(line)
            self.line_key_to_line_info.append((num_bytes_to_seek, num_bytes_to_read))
            num_bytes_to_seek += num_bytes_to_read
            line_idx += 1
            line = self.mmap_handle.readline()
        assert line_idx == len(self.line_key_to_line_info)

    def access_line(self, line_no):
        num_bytes_to_seek, num_bytes_to_read = self.line_key_to_line_info[line_no]
        self.mmap_handle.seek(num_bytes_to_seek, os.SEEK_SET)
        line = self.mmap_handle.read(num_bytes_to_read).decode("utf-8")
        json_line = json.loads(line)
        return json_line

    def length(self):
        return len(self.line_key_to_line_info)

    def __len__(self):
        return self.length()

    def __getitem__(self, item):
        return self.access_line(item)

    def __iter__(self):
        return iter(self.access_line(i) for i in range(self.length()))


class JSONLWithCkpts:
    def __init__(self, file_path):
        self.file_path = file_path
        self.existed_ids = set()
        self.fp = None

    def should_skip_line(self, line_id):
        return line_id in self.existed_ids

    def write_line(self, line_id, line_d):
        self.existed_ids.add(line_id)
        self.fp.write(
            "{}\n".format(
                json.dumps({"line_id": line_id, "data": line_d}, ensure_ascii=False)
            )
        )

    def __enter__(self):
        self.existed_ids.clear()
        if os.path.exists(self.file_path):
            with fopen(self.file_path) as fp:
                for i in fp:
                    try:
                        j = json.loads(i)
                        self.existed_ids.add(j["line_id"])
                    except json.JSONDecodeError as e:
                        logger.critical("Failed to parse line {}".format(i))
        self.fp = fopen(self.file_path, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.existed_ids.clear()
        self.fp.close()
        self.fp = None
