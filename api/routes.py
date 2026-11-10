"""The /v1 routes. Thin: parse and validate the request, call the Service, add caching headers."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, make_response, request

from api.cache import make_etag, matches
from api.errors import ApiError
from api.service import Service, scoring_param, season_param, slug_param

bp = Blueprint("v1", __name__, url_prefix="/v1")


def svc() -> Service:
    return current_app.extensions["service"]


def cached_json(body: dict, snap, max_age: int):
    """A JSON response with an ETag from the snapshot and the query, and a 304 when the client already has it."""
    etag = make_etag(snap.model_id, snap.season, snap.as_of, snap.created_at, request.full_path)
    if matches(request.headers.get("If-None-Match"), etag):
        resp = make_response("", 304)
    else:
        resp = make_response(jsonify(body))
    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


@bp.get("/models")
def models():
    return jsonify(svc().models())


@bp.get("/seasons")
def seasons():
    return jsonify(svc().seasons())


@bp.get("/predict/matchup")
def predict_matchup():
    a = slug_param("team_a", request.args.get("team_a"))
    b = slug_param("team_b", request.args.get("team_b"))
    season = season_param(request.args.get("season"))
    model = request.args.get("model") or None
    as_of = request.args.get("as_of") or None
    body, snap = svc().matchup(a, b, season, model, as_of)
    return cached_json(body, snap, svc().cache_age(snap))


@bp.get("/bracket/<int:season>")
def bracket(season):
    season_param(season)
    scoring = scoring_param(request.args.get("scoring"))
    body, snap = svc().bracket(season, request.args.get("model") or None, scoring)
    return cached_json(body, snap, svc().cache_age(snap))


@bp.get("/bracket/<int:season>/matchups")
def bracket_matchups(season):
    season_param(season)
    rnd = request.args.get("round", "1")
    if rnd != "1":
        raise ApiError("unsupported_round", "Only round 1 is known before results arrive. POST results to /v1/bracket/{season}/odds for later rounds.", 400, {"round": rnd})
    body, snap = svc().bracket(season, request.args.get("model") or None)
    out = {"season": season, "round": 1, "snapshot": body["snapshot"], "games": body["first_round"], "notes": body["notes"]}
    return cached_json(out, snap, svc().cache_age(snap))


@bp.post("/bracket/<int:season>/odds")
def bracket_odds(season):
    season_param(season)
    if request.content_length and request.content_length > current_app.config["MAX_BODY_BYTES"]:
        raise ApiError("too_large", "Request body is too large", 413)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("bad_body", "Send a JSON object like {\"results\": [{\"team_a\": ..., \"team_b\": ..., \"winner\": ...}]}", 400)
    results = data.get("results", [])
    if not isinstance(results, list) or len(results) > 63:
        raise ApiError("bad_body", "results must be a list of at most 63 games", 400)
    raw = data.get("scoring")
    scoring = scoring_param(",".join(str(x) for x in raw) if isinstance(raw, list) else raw)
    body, _ = svc().bracket(season, data.get("model") or None, scoring, results)
    resp = make_response(jsonify(body))
    resp.headers["Cache-Control"] = "no-store"
    return resp
