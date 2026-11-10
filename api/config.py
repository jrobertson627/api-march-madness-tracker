"""Settings, all from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"]


@dataclass
class Config:
    store: str = "auto"                                   # auto | postgres | files
    file_store_dir: Path = ROOT / "artifacts" / "predictions"
    cors_origins: list = field(default_factory=lambda: list(DEV_ORIGINS))
    cache_size: int = 256
    max_body_bytes: int = 20_000

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        cfg.store = os.environ.get("STORE", cfg.store)
        if os.environ.get("FILE_STORE_DIR"):
            cfg.file_store_dir = Path(os.environ["FILE_STORE_DIR"])
        origins = os.environ.get("CORS_ORIGINS")
        if origins:
            cfg.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        return cfg
