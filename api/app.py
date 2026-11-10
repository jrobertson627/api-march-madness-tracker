"""Application factory."""

from __future__ import annotations

import os
import time
import uuid

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from api import errors, legacy, routes
from api.config import Config
from api.service import Service
from api.store import FileStore, PostgresStore, Store


def build_store(cfg: Config) -> Store | None:
    kind = cfg.store
    if kind == "auto":
        kind = "postgres" if os.environ.get("DATABASE_URL") else ("files" if cfg.file_store_dir.exists() else "none")
    if kind == "postgres":
        return PostgresStore()
    if kind == "files":
        return FileStore(cfg.file_store_dir)
    return None


def create_app(config: Config | None = None, store: Store | None = None) -> Flask:
    cfg = config or Config.from_env()
    app = Flask(__name__)
    app.config["MAX_BODY_BYTES"] = cfg.max_body_bytes
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_body_bytes
    CORS(app, origins=cfg.cors_origins)
    errors.register(app)
    store = store if store is not None else build_store(cfg)
    app.extensions["service"] = Service(store, cfg.cache_size)

    app.register_blueprint(routes.bp)
    app.register_blueprint(legacy.bp)

    @app.get("/health")
    def health():
        return jsonify(app.extensions["service"].health())  # always 200 while the web service itself is up

    @app.before_request
    def start():
        g.t0, g.rid = time.perf_counter(), uuid.uuid4().hex[:8]

    @app.after_request
    def finish(resp):
        resp.headers["X-Request-Id"] = getattr(g, "rid", "-")
        app.logger.info("%s %s %s %.1fms rid=%s", request.method, request.path, resp.status_code, (time.perf_counter() - getattr(g, "t0", time.perf_counter())) * 1000, getattr(g, "rid", "-"))
        return resp

    return app
