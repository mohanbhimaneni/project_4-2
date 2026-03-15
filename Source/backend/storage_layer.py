from __future__ import annotations

import hashlib
import os
import struct
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

_MAGIC = b"SDST1"
_VERSION = 1
_HEADER_FMT = "<5sBIQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class StorageLayerError(Exception):
    """Raised when secure storage encode/decode fails."""


class _LFSR:
    """32-bit LFSR keystream generator used for stream XOR encryption."""

    POLYNOMIAL = 0x80000009

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next_bit(self) -> int:
        out = self.state & 1
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            self.state ^= self.POLYNOMIAL
        return out

    def next_byte(self) -> int:
        b = 0
        for _ in range(8):
            b = (b << 1) | self.next_bit()
        return b


def _derive_key(secret: str) -> int:
    digest = hashlib.sha256((secret + "::secure-storage").encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _xor_lfsr(data: bytes, seed: int) -> bytes:
    lfsr = _LFSR(seed)
    return bytes((b ^ lfsr.next_byte()) for b in data)


def _lzw_compress(data: bytes) -> bytes:
    if not data:
        return b""

    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256
    w = b""
    codes: list[int] = []

    for byte in data:
        wc = w + bytes([byte])
        if wc in dictionary:
            w = wc
        else:
            if w:
                codes.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = bytes([byte])

    if w:
        codes.append(dictionary[w])

    return b"".join(struct.pack("<I", c) for c in codes)


def _lzw_decompress(payload: bytes, expected_size: int) -> bytes:
    if not payload:
        return b""

    if len(payload) % 4 != 0:
        raise StorageLayerError("Invalid LZW payload length")

    codes = [struct.unpack("<I", payload[i : i + 4])[0] for i in range(0, len(payload), 4)]
    if not codes:
        return b""

    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256

    w = dictionary.get(codes[0], b"")
    out = bytearray(w)

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == next_code:
            entry = w + w[:1]
        else:
            raise StorageLayerError("Invalid LZW code stream")

        out.extend(entry)
        dictionary[next_code] = w + entry[:1]
        next_code += 1
        w = entry

    if expected_size and len(out) != expected_size:
        raise StorageLayerError(
            f"LZW size mismatch: expected {expected_size}, got {len(out)}"
        )

    return bytes(out)


def secure_encode(raw: bytes, secret: str) -> tuple[bytes, dict[str, Any]]:
    compressed = _lzw_compress(raw)
    seed_raw = int.from_bytes(os.urandom(4), "little")
    stream_seed = seed_raw ^ _derive_key(secret)
    cipher = _xor_lfsr(compressed, stream_seed)

    header = struct.pack(_HEADER_FMT, _MAGIC, _VERSION, seed_raw, len(raw), len(compressed))
    blob = header + cipher

    ratio = (len(compressed) / len(raw)) if raw else 1.0
    meta = {
        "algorithm": "LZW+LFSR",
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "stored_bytes": len(blob),
        "compression_ratio": round(ratio, 6),
        "encrypted": True,
    }
    return blob, meta


def secure_decode(blob: bytes, secret: str) -> bytes:
    # Backward compatibility: non-secure legacy files are returned as-is.
    if len(blob) < _HEADER_SIZE:
        return blob

    magic, version, seed_raw, expected_raw, compressed_size = struct.unpack(
        _HEADER_FMT, blob[:_HEADER_SIZE]
    )
    if magic != _MAGIC or version != _VERSION:
        return blob

    cipher = blob[_HEADER_SIZE:]
    if len(cipher) != compressed_size:
        raise StorageLayerError("Cipher payload size mismatch")

    stream_seed = seed_raw ^ _derive_key(secret)
    compressed = _xor_lfsr(cipher, stream_seed)
    return _lzw_decompress(compressed, expected_raw)


def secure_write_bytes(path: Path, raw: bytes, secret: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob, meta = secure_encode(raw, secret)
    path.write_bytes(blob)
    return meta


def secure_read_bytes(path: str | Path, secret: str) -> bytes:
    raw = Path(path).read_bytes()
    return secure_decode(raw, secret)


def secure_write_numpy(path: Path, arr: np.ndarray, secret: str) -> dict[str, Any]:
    buf = BytesIO()
    np.save(buf, arr)
    return secure_write_bytes(path, buf.getvalue(), secret)


def secure_read_numpy(path: str | Path, secret: str) -> np.ndarray:
    raw = secure_read_bytes(path, secret)
    return np.load(BytesIO(raw), allow_pickle=False)


def secure_read_dicom(path: str | Path, secret: str) -> pydicom.dataset.FileDataset:
    raw = secure_read_bytes(path, secret)
    return pydicom.dcmread(BytesIO(raw))
