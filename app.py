import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from data import teams

app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/teams")
def get_teams():
    return jsonify(teams.list_teams())


@app.route("/teams/<team_id>/stats")
def get_team_stats(team_id):
    season = request.args.get("season", type=int) or teams.SEASON
    team = teams.get_team_stats(team_id, season)
    if team is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(team)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
