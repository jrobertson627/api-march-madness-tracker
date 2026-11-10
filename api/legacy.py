"""The original team routes, unchanged in behavior. They read the scraper's tables directly."""

from flask import Blueprint, jsonify, request

bp = Blueprint("legacy", __name__)


@bp.get("/teams")
def get_teams():
    from data import teams

    season = request.args.get("season", type=int) or teams.SEASON
    return jsonify(teams.list_teams(season))


@bp.get("/teams/<team_id>/stats")
def get_team_stats(team_id):
    from data import teams

    season = request.args.get("season", type=int) or teams.SEASON
    team = teams.get_team_stats(team_id, season)
    if team is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(team)
