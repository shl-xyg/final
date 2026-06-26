"""Small compatibility shim for CALVIN's optional ``pyhash`` dependency.

The upstream ``pyhash`` package is no longer installable on current Python
versions because its setup metadata still references ``use_2to3``. CALVIN only
needs ``fnv1_32`` as a deterministic callable hash, so we provide that API
locally and keep the evaluation environment reproducible.
"""


def fnv1_32():
    """Return a callable implementing 32-bit FNV-1a over UTF-8 text."""

    def _hash(value):
        data = str(value).encode("utf-8")
        hashed = 0x811C9DC5
        for byte in data:
            hashed ^= byte
            hashed = (hashed * 0x01000193) & 0xFFFFFFFF
        return hashed

    return _hash
