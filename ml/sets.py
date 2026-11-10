"""Named experiment sets.

  baselines  Fixed models with nothing to select, run on all five seasons.
  models     Anything tuned. Leaves the locked-box season (2026) out of training
             and testing entirely. Reference baselines are repeated on the same
             seasons so comparisons are like for like.
"""

from __future__ import annotations

from ml.experiment import Config

LOCKED = (2026,)
C_GRID = [0.01, 0.1, 1.0]
MIX_GRID = ["equal", "tourney_x5", "tourney_x20"]
HGB_GRID = {"max_depth": [2, 3], "max_iter": [100, 200], "mix": ["equal", "tourney_x5"]}


def baselines(exclude=()) -> list[Config]:
    tag = " (select seasons)" if exclude else ""
    out = [
        Config(f"B0 coin flip{tag}", "coin", "rating_core", exclude_seasons=exclude),
        Config(f"B1 seed logistic{tag}", "logistic", "seed_only", mix="ncaa_only", exclude_seasons=exclude),
        Config(f"B2 seed pairs{tag}", "seed_pair", "seed_pair", mix="ncaa_only", exclude_seasons=exclude),
        Config(f"B3 elo probability{tag}", "elo_prob", "elo_prob", exclude_seasons=exclude),
        Config(f"B4 efficiency gap, all rows{tag}", "logistic", "rating_core", mix="equal", exclude_seasons=exclude),
        Config(f"B4 efficiency gap, ncaa only{tag}", "logistic", "rating_core", mix="ncaa_only", exclude_seasons=exclude),
    ]
    if not exclude:
        out += [Config(f"{c.name} [forward]", c.model, c.feature_set, mix=c.mix, protocol="forward") for c in out[1:]]
    return out


def models() -> list[Config]:
    ex = LOCKED
    return [
        *baselines(ex),
        Config("M5 logistic ratings_plus", "logistic", "ratings_plus", grid={"C": C_GRID, "mix": MIX_GRID}, exclude_seasons=ex),
        Config("M5 logistic full", "logistic", "full", grid={"C": C_GRID, "mix": MIX_GRID}, exclude_seasons=ex),
        Config("M5 logistic ratings_plus + platt", "logistic", "ratings_plus", grid={"C": C_GRID, "mix": MIX_GRID}, calibration="platt", exclude_seasons=ex),
        Config("M5 logistic full_with_seed", "logistic", "full_with_seed", grid={"C": C_GRID, "mix": MIX_GRID}, exclude_seasons=ex),
        Config("M6 hgb ratings_plus", "hgb", "ratings_plus", grid=HGB_GRID, exclude_seasons=ex),
        Config("M6 hgb full", "hgb", "full", grid=HGB_GRID, exclude_seasons=ex),
        Config("M6 hgb full monotone", "hgb", "full", params={"monotone": True}, grid=HGB_GRID, exclude_seasons=ex),
        Config("S1 ridge scores + logistic ratings_plus", "logistic", "ratings_plus", grid={"C": C_GRID}, score_model="ridge", exclude_seasons=ex),
        Config("S2 hgb scores + logistic ratings_plus", "logistic", "ratings_plus", grid={"C": C_GRID}, score_model="hgb", exclude_seasons=ex),
    ]


def followups() -> list[Config]:
    """Run after `models`: isotonic versus Platt calibration, and forward-chaining checks of the leaders.

    Forward chaining with 2026 left out tests only 2024 and 2025 (about 134 games), so compare these rows
    with each other, not with leave-one-season-out rows.
    """
    ex = LOCKED
    return [
        Config("M5 logistic ratings_plus + isotonic", "logistic", "ratings_plus", grid={"C": C_GRID, "mix": MIX_GRID}, calibration="isotonic", exclude_seasons=ex),
        Config("M6 hgb full + platt", "hgb", "full", grid=HGB_GRID, calibration="platt", exclude_seasons=ex),
        Config("M6 hgb full + isotonic", "hgb", "full", grid=HGB_GRID, calibration="isotonic", exclude_seasons=ex),
        Config("F B4 efficiency gap [forward, select seasons]", "logistic", "rating_core", protocol="forward", exclude_seasons=ex),
        Config("F M5 logistic full [forward, select seasons]", "logistic", "full", grid={"C": C_GRID, "mix": MIX_GRID}, protocol="forward", exclude_seasons=ex),
        Config("F M6 hgb full [forward, select seasons]", "hgb", "full", grid=HGB_GRID, protocol="forward", exclude_seasons=ex),
    ]


FINAL_NAME = "FINAL boosted full monotone + ridge scores"


def final() -> list[Config]:
    """The design chosen from the 2022 to 2025 results (decision p2-25): boosted trees on the full feature set,
    monotone in the rating differences, no calibration, ridge margin and total models for scores.
    Boosted full monotone had the lowest selection-season log loss, 0.5562, a tie with the plain version (0.5564)."""
    return [Config(FINAL_NAME, "hgb", "full", params={"monotone": True}, grid=HGB_GRID, score_model="ridge")]


SETS = {"baselines": baselines, "models": models, "followups": followups, "final": final}


def find(name: str) -> Config:
    """Look a config up by exact name across all sets."""
    for build in SETS.values():
        for cfg in build():
            if cfg.name == name:
                return cfg
    raise KeyError(f"no experiment named {name!r}")
