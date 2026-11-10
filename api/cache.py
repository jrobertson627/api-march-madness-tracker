"""HTTP caching helpers and a small in-process cache.

Published snapshots never change, so the same request gets the same answer for the life of a snapshot. The ETag is
built from what identifies the answer, and a matching If-None-Match gets a 304 with no body.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict


class LRU:
    """A tiny thread-safe least-recently-used cache."""

    def __init__(self, maxsize: int = 256):
        self.maxsize, self._d, self._lock = maxsize, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)

    def clear(self):
        with self._lock:
            self._d.clear()


def make_etag(*parts) -> str:
    return '"' + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20] + '"'


def matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    return etag in [t.strip() for t in if_none_match.split(",")] or if_none_match.strip() == "*"
