"""Turn a table of betting lines (BetIQ's layout) into per-game benchmark rows.

You supply the file. Nothing here fetches data from any site. The expected columns are the ones BetIQ's game table
shows, one row per team per game:

    Date, Team, Opponent, Location, Spread, Total (O/U), Money Line, Score

`Spread` is the team's own spread (negative when it is the favorite), `Money Line` is American odds for that team,
and `Score` is "team-opponent". Extra columns are ignored, headers are matched ignoring case, and blank, "-" or
"NA" cells become missing.

Output is one row per game in the matrix's canonical order (team_a is the smaller team id), with the closing
spread, total, both moneylines and two win probabilities: `p_ml` from the de-vigged moneylines and `p_spread` from
the spread. `p_a_win` is `p_ml` when both moneylines exist and `p_spread` otherwise. Feed the result to
`ml.benchmarks.load_benchmark` / `compare_to_model`, after `match_to_matrix` if dates may differ by a day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.benchmarks import estimate_sigma, spread_to_prob
from ml.team_names import ALIASES, match_names, normalize_name

# BetIQ's short names -> school names inside our team ids (see ml.team_names.slug_name). Only names whose
# school exists in the database are listed; others come back as unmatched.
BETIQ_ALIASES_RAW = {
    "AR Lit Rock": "arkansas little rock", "Little Rock": "arkansas little rock", "AR-Pine Bluff": "arkansas pine bluff",
    "Abl Christian": "abilene christian", "Albany": "albany ny", "App State": "appalachian state",
    "Bethune": "bethune cookman", "Boston U": "boston university", "Bowling Green": "bowling green state",
    "C Arkansas": "central arkansas", "C Connecticut": "central connecticut state", "C Michigan": "central michigan",
    "CS Bakersfield": "cal state bakersfield", "CS Fullerton": "cal state fullerton", "CS Northridge": "cal state northridge",
    "Cal Baptist": "california baptist", "Charleston": "college of charleston", "Charleston So": "charleston southern",
    "Coastal Car": "coastal carolina", "E Carolina": "east carolina", "E Illinois": "eastern illinois",
    "E Kentucky": "eastern kentucky", "E Michigan": "eastern michigan", "E Tennessee St": "east tennessee state",
    "E Washington": "eastern washington", "F Dickinson": "fairleigh dickinson", "FGCU": "florida gulf coast",
    "Florida Intl": "florida international", "G Washington": "george washington", "Georgia So": "georgia southern",
    "Hou Christian": "houston baptist", "Houston Bap": "houston baptist", "IU Indy": "iupui", "J Madison": "james madison",
    "Kansas City": "missouri kansas city", "LIU": "long island university", "LIU-Brooklyn": "long island university",
    "Loyola Chi": "loyola il", "Loyola Mymt": "loyola marymount", "Maryland ES": "maryland eastern shore",
    "McNeese": "mcneese state", "Miami": "miami fl", "Middle Tenn": "middle tennessee", "Miss Valley St": "mississippi valley state",
    "Mt St Mary's": "mount st marys", "N Alabama": "north alabama", "N Arizona": "northern arizona",
    "N Colorado": "northern colorado", "N Dakota St": "north dakota state", "N Florida": "north florida",
    "N Illinois": "northern illinois", "N Iowa": "northern iowa", "N Kentucky": "northern kentucky", "N Texas": "north texas",
    "NC A&T": "north carolina at", "NC Asheville": "north carolina asheville", "NC Central": "north carolina central",
    "NC Greensboro": "north carolina greensboro", "NC Wilmington": "north carolina wilmington", "NW State": "northwestern state",
    "Nicholls": "nicholls state", "Omaha": "nebraska omaha", "Penn": "pennsylvania", "Queens": "queens nc",
    "S Alabama": "south alabama", "S Carolina St": "south carolina state", "S Dakota St": "south dakota state",
    "S Florida": "south florida", "S Illinois": "southern illinois", "S Indiana": "southern indiana", "S Utah": "southern utah",
    "SC Upstate": "south carolina upstate", "SE Louisiana": "southeastern louisiana", "SE Missouri St": "southeast missouri state",
    "SF Austin": "stephen f austin", "SIU Edward": "southern illinois edwardsville", "Sam Hous St": "sam houston state",
    "Sam Houston": "sam houston state", "St Thomas": "st thomas mn", "TX A&M-Com": "texas am commerce",
    "TX-Pan Am": "texas pan american", "UT Rio Grande": "texas pan american", "Tenn Tech": "tennessee tech",
    "Texas A&M-CC": "texas am corpus christi", "Texas So": "texas southern", "UC Davis": "california davis",
    "UC Riverside": "california riverside", "UCSB": "california santa barbara", "UCSD": "california san diego",
    "UL Monroe": "louisiana monroe", "UMass Lowell": "massachusetts lowell", "Utah Tech": "dixie state",
    "VMI": "virginia military institute", "W Carolina": "western carolina", "W Georgia": "west georgia",
    "W Illinois": "western illinois", "W Kentucky": "western kentucky", "W Michigan": "western michigan",
    "WI-Grn Bay": "green bay", "WI-Milwkee": "milwaukee",
}
BETIQ_ALIASES = {normalize_name(k): normalize_name(v) for k, v in BETIQ_ALIASES_RAW.items()}

COLUMNS = {"date": "game_date", "team": "team", "opponent": "opp", "location": "location", "spread": "spread",
           "total (o/u)": "total", "total": "total", "money line": "ml", "moneyline": "ml", "score": "score"}
MISSING = {"", "-", "--", "na", "n/a", "nan", "none", "pk"}
SIGMA_FALLBACK = 11.0


def season_of(dates: pd.Series) -> pd.Series:
    """Season (ending year) from a game date: August and later belong to the next year's season."""
    d = pd.to_datetime(dates)
    return d.dt.year + (d.dt.month >= 8).astype(int)


def _num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace("+", "", regex=False)
    return pd.to_numeric(s.where(~s.str.lower().isin(MISSING)), errors="coerce")


def ml_to_prob(ml) -> np.ndarray:
    """Implied probability from American odds (before removing the bookmaker's margin)."""
    ml = np.asarray(ml, dtype=float)
    ml = np.where((ml > -100) & (ml < 100), np.nan, ml)  # American odds are never between -100 and +100
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def devig(p_a, p_b) -> np.ndarray:
    """Win probability for A after removing the margin by normalizing the two implied probabilities."""
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    return p_a / (p_a + p_b)


def read_lines(df_or_path) -> pd.DataFrame:
    df = pd.read_csv(df_or_path, dtype=str) if not isinstance(df_or_path, pd.DataFrame) else df_or_path.copy()
    ren = {c: COLUMNS[c.strip().lower()] for c in df.columns if c.strip().lower() in COLUMNS}
    df = df.rename(columns=ren)
    need = {"game_date", "team", "opp"}
    if not need <= set(df.columns):
        raise ValueError(f"lines file needs columns Date, Team and Opponent (found {sorted(df.columns)})")
    out = pd.DataFrame({"game_date": pd.to_datetime(df["game_date"]), "team": df["team"].astype(str).str.strip(), "opp": df["opp"].astype(str).str.strip()})
    for c in ("spread", "total", "ml"):
        out[c] = _num(df[c]) if c in df else np.nan
    if "score" in df:
        parts = df["score"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
        out["pts_team"], out["pts_opp"] = pd.to_numeric(parts[0]), pd.to_numeric(parts[1])
    else:
        out["pts_team"] = out["pts_opp"] = np.nan
    return out


def betiq_to_games(df_or_path, team_table: pd.DataFrame, sigma: float | None = None, exclude_sigma_seasons=(),
                   aliases: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per game in canonical order, plus a report of what could not be matched.

    `sigma` converts spreads to probabilities. If None it is estimated from the file's own games in seasons not in
    `exclude_sigma_seasons`, so pass the seasons you are evaluating to keep the estimate out of the test.
    """
    rows = read_lines(df_or_path)
    mapping, unmatched = match_names(pd.concat([rows["team"], rows["opp"]]), team_table, ALIASES | BETIQ_ALIASES | (aliases or {}))
    rows["team_id"], rows["opp_id"] = rows["team"].map(mapping), rows["opp"].map(mapping)
    kept = rows.dropna(subset=["team_id", "opp_id"]).copy()
    dropped = len(rows) - len(kept)

    kept["team_a"] = np.where(kept["team_id"] < kept["opp_id"], kept["team_id"], kept["opp_id"])
    kept["team_b"] = np.where(kept["team_id"] < kept["opp_id"], kept["opp_id"], kept["team_id"])
    is_a = (kept["team_id"] == kept["team_a"]).to_numpy()
    kept["season"] = season_of(kept["game_date"])
    kept["spread_a"] = np.where(is_a, -kept["spread"], kept["spread"])  # positive: A is favored
    kept["ml_a"] = np.where(is_a, kept["ml"], np.nan)
    kept["ml_b"] = np.where(~is_a, kept["ml"], np.nan)
    kept["score_a"] = np.where(is_a, kept["pts_team"], kept["pts_opp"])
    kept["score_b"] = np.where(is_a, kept["pts_opp"], kept["pts_team"])

    key = ["season", "game_date", "team_a", "team_b"]
    g = kept.groupby(key, as_index=False).agg(
        spread_a=("spread_a", "first"), total=("total", "first"), ml_a=("ml_a", "first"), ml_b=("ml_b", "first"),
        score_a=("score_a", "first"), score_b=("score_b", "first"),
    )
    g["margin_a"] = g["score_a"] - g["score_b"]
    g["p_ml"] = devig(ml_to_prob(g["ml_a"]), ml_to_prob(g["ml_b"]))

    pool = g[~g["season"].isin(list(exclude_sigma_seasons)) & g["spread_a"].notna() & g["margin_a"].notna()]
    sigma_source = "given"
    if sigma is None:
        if len(pool) >= 200:
            sigma, sigma_source = estimate_sigma(pool["spread_a"], pool["margin_a"]), f"estimated from {len(pool)} games"
        else:
            sigma, sigma_source = SIGMA_FALLBACK, "fallback (too few games to estimate)"
    g["p_spread"] = spread_to_prob(g["spread_a"].fillna(np.nan), sigma)
    g["p_a_win"] = g["p_ml"].where(g["p_ml"].notna(), g["p_spread"])
    report = {"rows": len(rows), "rows_dropped_unmatched": dropped, "unmatched_names": unmatched, "games": len(g),
              "games_without_any_line": int(g["p_a_win"].isna().sum()), "sigma": float(sigma), "sigma_source": sigma_source}
    return g.reset_index(drop=True), report


def match_to_matrix(lines: pd.DataFrame, matrix: pd.DataFrame, tolerance_days: int = 1) -> pd.DataFrame:
    """Attach each matrix game's line, allowing the date to differ by up to `tolerance_days` (late tip-offs).

    Returns the matrix rows (any game_group) that found a line, with the line's columns and `line_date`.
    A pair that meets twice in a season takes the nearest date.
    """
    cols = ["season", "team_a", "team_b", "game_date", "p_a_win", "p_ml", "p_spread", "spread_a", "total"]
    m = matrix[["game_id", "season", "team_a", "team_b", "game_date"]].merge(
        lines[[c for c in cols if c in lines]].rename(columns={"game_date": "line_date"}), on=["season", "team_a", "team_b"], how="inner"
    )
    m["gap"] = (m["game_date"] - m["line_date"]).abs().dt.days
    m = m[m["gap"] <= tolerance_days].sort_values("gap").drop_duplicates("game_id")
    return m.drop(columns="gap").reset_index(drop=True)
