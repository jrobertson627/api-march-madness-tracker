"""Response shapes as JSON Schema. Tests validate real responses against these, and docs/api.md is generated from them."""

NUM = {"type": ["number", "null"]}
STR = {"type": "string"}
SNAPSHOT = {
    "type": "object", "required": ["model_id", "season", "as_of", "kind", "in_sample", "n_teams"],
    "properties": {"model_id": STR, "season": {"type": "integer"}, "as_of": STR, "kind": {"enum": ["selection_sunday", "latest"]},
                   "in_sample": {"type": "boolean"}, "n_teams": {"type": "integer"}},
}
MODEL = {
    "type": "object", "required": ["id", "name", "kind", "trained_seasons", "active"],
    "properties": {"id": STR, "name": STR, "kind": {"enum": ["live", "backtest"]}, "trained_seasons": {"type": "array", "items": {"type": "integer"}},
                   "data_hash": STR, "created_at": STR, "metrics": {"type": "object"}, "active": {"type": "boolean"}},
}
TEAM_REF = {"type": "object", "required": ["id", "name"], "properties": {"id": STR, "name": {"type": ["string", "null"]}}}
FACTOR = {"type": "object", "required": ["group", "effect"], "properties": {"group": STR, "effect": NUM, "headline": {"type": ["string", "null"]}, "a": NUM, "b": NUM}}
PICK = {"type": "object", "required": ["round", "round_name", "game", "team_a", "team_b", "pick", "p_pick_wins_game", "score_a", "score_b", "margin_a"],
        "properties": {"round": {"type": "integer"}, "round_name": STR, "game": {"type": "integer"}, "team_a": STR, "team_b": STR, "pick": STR, "p_pick_wins_game": NUM,
                       "score_a": NUM, "score_b": NUM, "margin_a": NUM}}
PLAYED = {"type": "object", "required": ["round", "round_name", "game", "team_a", "team_b", "winner"],
          "properties": {"round": {"type": "integer"}, "round_name": STR, "game": {"type": "integer"}, "team_a": STR, "team_b": STR, "winner": STR}}
TEAM_ROW = {"type": "object", "required": ["id", "name", "seed", "region"],
            "properties": {"id": STR, "name": {"type": ["string", "null"]}, "seed": {"type": ["integer", "null"]}, "region": {"type": ["string", "null"]}}}
ADV_ROW = {"type": "object", "required": ["season", "method", "log_loss", "brier", "events"],
           "properties": {"season": {"type": ["integer", "string"]}, "method": STR, "log_loss": NUM, "brier": NUM, "events": {"type": "integer"}}}
GAME_ROW = {"type": "object", "required": ["season", "method", "n", "log_loss", "brier", "accuracy"],
            "properties": {"season": {"type": ["integer", "string"]}, "method": STR, "n": {"type": "integer"}, "log_loss": NUM, "brier": NUM, "accuracy": NUM}}
PICK_ROW = {"type": "object", "required": ["season", "strategy", "realized", "expected"],
            "properties": {"season": {"type": "integer"}, "strategy": STR, "realized": NUM, "expected": NUM}}
PAIRED_ROW = {"type": "object", "required": ["candidate", "baseline", "games", "diff", "low", "high", "seasons_ahead", "seasons", "interval_excludes_zero"],
              "properties": {"candidate": STR, "baseline": STR, "games": {"type": "integer"}, "diff": NUM, "low": NUM, "high": NUM,
                             "seasons_ahead": {"type": "integer"}, "seasons": {"type": "integer"}, "interval_excludes_zero": {"type": "boolean"}}}
FLAG_ROW = {"type": "object", "required": ["season", "region", "favorite", "underdog", "favorite_seed", "underdog_seed", "p_underdog_model", "model_picks_upset", "model_above_history", "upset"],
            "properties": {"season": {"type": "integer"}, "region": STR, "favorite": STR, "favorite_name": {"type": ["string", "null"]}, "underdog": STR,
                           "underdog_name": {"type": ["string", "null"]}, "favorite_seed": {"type": "integer"}, "underdog_seed": {"type": "integer"},
                           "p_underdog_model": NUM, "model_picks_upset": {"type": "boolean"}, "model_above_history": {"type": "boolean"}, "upset": {"type": "boolean"}}}
SHARE = {"type": "object", "required": ["games", "upsets"], "properties": {"games": {"type": "integer"}, "upsets": {"type": "integer"}}}
FIELD_TEAM = {"type": "object", "required": ["team", "name", "seed", "region", "p_champion"],
              "properties": {"team": STR, "name": {"type": ["string", "null"]}, "seed": {"type": "integer"}, "region": STR,
                             "p_round_of_64": NUM, "p_round_of_32": NUM, "p_sweet_16": NUM, "p_elite_8": NUM, "p_final_four": NUM,
                             "p_championship_game": NUM, "p_champion": NUM}}
FIRST_ROUND = {"type": "object", "required": ["region", "favorite", "underdog", "p_underdog_model", "model_picks_upset", "flagged"],
               "properties": {"region": STR, "favorite": STR, "underdog": STR, "favorite_seed": {"type": "integer"}, "underdog_seed": {"type": "integer"},
                              "p_underdog_model": NUM, "p_underdog_history": NUM, "model_picks_upset": {"type": "boolean"},
                              "model_above_history": {"type": "boolean"}, "flagged": {"type": "boolean"}}}

SCHEMAS = {
    "matchup": {
        "type": "object", "required": ["team_a", "team_b", "p_a_win", "p_b_win", "projected", "factors", "model", "snapshot", "notes"],
        "properties": {
            "team_a": TEAM_REF, "team_b": TEAM_REF, "p_a_win": NUM, "p_b_win": NUM,
            "projected": {"type": "object", "required": ["score_a", "score_b", "margin_a", "total", "basis"],
                          "properties": {"score_a": NUM, "score_b": NUM, "margin_a": NUM, "total": NUM, "basis": STR}},
            "factors": {"type": "array", "items": FACTOR}, "interaction": NUM, "model": MODEL, "snapshot": SNAPSHOT,
            "notes": {"type": "array", "items": STR}},
    },
    "bracket": {
        "type": "object", "required": ["season", "snapshot", "scoring", "locked_results", "field", "picks", "expected_points", "first_round", "results", "notes"],
        "properties": {"season": {"type": "integer"}, "snapshot": SNAPSHOT, "scoring": {"type": "array", "items": {"type": "integer"}},
                       "locked_results": {"type": "integer"}, "field": {"type": "array", "items": FIELD_TEAM},
                       "results": {"type": "array", "items": PLAYED},
                       "picks": {"type": "object", "required": ["best_expected_points", "model_favorite", "seed_chalk"],
                                 "additionalProperties": {"type": "array", "items": PICK}},
                       "expected_points": {"type": "object", "additionalProperties": NUM},
                       "first_round": {"type": "array", "items": FIRST_ROUND}, "notes": {"type": "array", "items": STR}},
    },
    "round_matchups": {
        "type": "object", "required": ["season", "round", "snapshot", "games", "notes"],
        "properties": {"season": {"type": "integer"}, "round": {"type": "integer"}, "snapshot": SNAPSHOT, "games": {"type": "array", "items": FIRST_ROUND}, "notes": {"type": "array"}},
    },
    "teams": {"type": "object", "required": ["season", "snapshot", "teams", "notes"],
              "properties": {"season": {"type": "integer"}, "snapshot": SNAPSHOT, "teams": {"type": "array", "items": TEAM_ROW}, "notes": {"type": "array", "items": STR}}},
    "backtest": {
        "type": "object", "required": ["model_id", "data_hash", "created_at", "selection", "locked", "limits"],
        "properties": {
            "model_id": STR, "data_hash": STR, "created_at": STR, "limits": {"type": "array", "items": STR}, "notes": {"type": "array", "items": STR},
            "selection": {"type": "object", "required": ["seasons", "advancement", "games_scored", "picks", "upset_flags", "upset_summary", "paired"],
                          "properties": {"seasons": {"type": "array", "items": {"type": "integer"}}, "advancement": {"type": "array", "items": ADV_ROW},
                                         "games_scored": {"type": "array", "items": GAME_ROW}, "picks": {"type": "array", "items": PICK_ROW},
                                         "upset_flags": {"type": "array", "items": FLAG_ROW},
                                         "upset_summary": {"type": "object", "required": ["all_games", "model_picks_upset", "model_above_history", "no_flag"],
                                                           "properties": {"all_games": SHARE, "model_picks_upset": SHARE, "model_above_history": SHARE, "no_flag": SHARE}},
                                         "paired": {"type": "array", "items": PAIRED_ROW}}},
            "locked": {"oneOf": [{"type": "null"}, {"type": "object", "required": ["season", "advancement", "games_scored", "picks", "paired"],
                                                    "properties": {"season": {"type": "integer"}, "advancement": {"type": "array", "items": ADV_ROW},
                                                                   "games_scored": {"type": "array", "items": GAME_ROW}, "picks": {"type": "array", "items": PICK_ROW},
                                                                   "paired": {"type": "array", "items": PAIRED_ROW}}}]},
        },
    },
    "models": {"type": "object", "required": ["models"], "properties": {"models": {"type": "array", "items": MODEL}}},
    "seasons": {"type": "object", "required": ["seasons"], "properties": {"seasons": {"type": "array", "items": {
        "type": "object", "required": ["season", "snapshots"], "properties": {"season": {"type": "integer"}, "snapshots": {"type": "array", "items": SNAPSHOT}}}}}},
    "health": {"type": "object", "required": ["status", "store"], "properties": {"status": {"enum": ["ok", "degraded"]}, "store": {"type": "boolean"},
                                                                                  "active_model": {"oneOf": [MODEL, {"type": "null"}]}, "latest_snapshot": {"oneOf": [SNAPSHOT, {"type": "null"}]}}},
    "error": {"type": "object", "required": ["error"], "properties": {"error": {"type": "object", "required": ["code", "message", "details"],
                                                                                   "properties": {"code": STR, "message": STR, "details": {"type": "object"}}}}},
}
