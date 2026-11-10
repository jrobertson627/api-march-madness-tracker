"""One error shape for the whole API: {"error": {"code", "message", "details"}}."""

from __future__ import annotations

from flask import jsonify
from werkzeug.exceptions import HTTPException


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details or {}

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def register(app) -> None:
    @app.errorhandler(ApiError)
    def api_error(e: ApiError):
        return jsonify(e.body()), e.status

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 413: "too_large", 415: "unsupported_media_type"}.get(e.code, "http_error")
        return jsonify(ApiError(code, e.description or e.name, e.code or 500).body()), e.code or 500

    @app.errorhandler(Exception)
    def unexpected(e: Exception):
        app.logger.exception("unhandled error")
        return jsonify(ApiError("internal_error", "Something went wrong on the server.", 500).body()), 500
