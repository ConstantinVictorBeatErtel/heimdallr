"""Pure-Python TFRecord reader (no TensorFlow runtime required)."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from wildfire.data.constants import INPUT_FEATURES, OUTPUT_FEATURES, PATCH_SIZE
from wildfire.data.proto import example_pb2

FEATURE_KEYS = INPUT_FEATURES + OUTPUT_FEATURES
_RECORD_EXPECTED_FLOATS = PATCH_SIZE * PATCH_SIZE


def _decode_record(raw: bytes) -> bytes:
    """Validate TFRecord CRC framing and return the payload bytes."""
    if len(raw) < 16:
        msg = "TFRecord frame is too short"
        raise ValueError(msg)

    length = struct.unpack("<Q", raw[:8])[0]
    length_crc = struct.unpack("<I", raw[8:12])[0]
    payload = raw[12 : 12 + length]
    data_crc = struct.unpack("<I", raw[12 + length : 16 + length])[0]

    if length != len(payload):
        msg = "TFRecord length header does not match payload size"
        raise ValueError(msg)

    if length_crc != zlib.crc32(raw[:8]) & 0xFFFFFFFF:
        msg = "TFRecord length CRC check failed"
        raise ValueError(msg)

    if data_crc != zlib.crc32(payload) & 0xFFFFFFFF:
        msg = "TFRecord data CRC check failed"
        raise ValueError(msg)

    return payload


def iter_tfrecord_bytes(path: Path) -> Iterator[bytes]:
    """Yield raw Example payloads from one TFRecord shard."""
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) < 8:
                msg = f"Unexpected EOF while reading {path}"
                raise EOFError(msg)

            (length,) = struct.unpack("<Q", header)
            rest = handle.read(4 + length + 4)
            if len(rest) < 4 + length + 4:
                msg = f"Unexpected EOF while reading record body in {path}"
                raise EOFError(msg)

            yield _decode_record(header + rest)


def parse_tfrecord_bytes(serialized: bytes) -> dict[str, np.ndarray]:
    """Parse one TFRecord Example into a dict of (64, 64) numpy arrays."""
    example = example_pb2.Example()
    example.ParseFromString(serialized)

    parsed: dict[str, np.ndarray] = {}
    for key in FEATURE_KEYS:
        if key not in example.features.feature:
            msg = f"Missing feature {key!r} in TFRecord example"
            raise KeyError(msg)

        values = list(example.features.feature[key].float_list.value)
        if len(values) != _RECORD_EXPECTED_FLOATS:
            msg = (
                f"Feature {key!r} expected {_RECORD_EXPECTED_FLOATS} values, "
                f"got {len(values)}"
            )
            raise ValueError(msg)

        parsed[key] = np.asarray(values, dtype=np.float32).reshape(
            PATCH_SIZE,
            PATCH_SIZE,
        )

    return parsed


def iter_tfrecord_file(path: Path) -> Iterator[dict[str, np.ndarray]]:
    """Yield parsed examples from a single TFRecord shard."""
    for payload in iter_tfrecord_bytes(path):
        yield parse_tfrecord_bytes(payload)


def iter_tfrecord_files(paths: list[Path]) -> Iterator[dict[str, np.ndarray]]:
    """Yield parsed examples from multiple TFRecord shards."""
    for path in paths:
        yield from iter_tfrecord_file(path)


def build_record_index(paths: list[Path]) -> list[tuple[int, int]]:
    """Build a flat index mapping sample_idx -> (file_idx, record_idx_in_file)."""
    index: list[tuple[int, int]] = []
    for file_idx, path in enumerate(paths):
        for record_idx, _ in enumerate(iter_tfrecord_bytes(path)):
            index.append((file_idx, record_idx))
    return index


def read_record_at(
    paths: list[Path],
    file_idx: int,
    record_idx: int,
) -> dict[str, np.ndarray]:
    """Read one specific record by shard index and position within the shard."""
    path = paths[file_idx]
    for idx, payload in enumerate(iter_tfrecord_bytes(path)):
        if idx == record_idx:
            return parse_tfrecord_bytes(payload)
    msg = f"Record {record_idx} not found in {path}"
    raise IndexError(msg)


def write_tfrecord_example(path: Path, features: dict[str, np.ndarray]) -> None:
    """Write one Example to a TFRecord file (used by tests and conversion scripts)."""
    example = example_pb2.Example()
    for key, array in features.items():
        flat = np.asarray(array, dtype=np.float32).reshape(-1)
        example.features.feature[key].float_list.value.extend(flat.tolist())

    payload = example.SerializeToString()
    length = struct.pack("<Q", len(payload))
    length_crc = struct.pack("<I", zlib.crc32(length) & 0xFFFFFFFF)
    data_crc = struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)

    with path.open("ab") as handle:
        handle.write(length)
        handle.write(length_crc)
        handle.write(payload)
        handle.write(data_crc)
