"""Match school names (as betting lines write them) to the matrix's team ids.

The matrix uses Sports Reference ids such as `www.sports-reference.com/cbb/schools/duke/men`.
Lines use names such as "Duke", "UConn" or "Michigan St.". `match_names` normalizes both sides, applies a short
alias table, and returns what it could not match, so no game is dropped silently. Extend `ALIASES` or pass your own.
"""

from __future__ import annotations

import re

import pandas as pd

ALIASES = {  # normalized line name -> normalized Sports Reference display name
    "uconn": "connecticut",
    "unc": "north carolina",
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "ucf": "central florida",
    "lsu": "louisiana state",
    "vcu": "virginia commonwealth",
    "usc": "southern california",
    "umass": "massachusetts",
    "unlv": "nevada las vegas",
    "uab": "alabama birmingham",
    "utep": "texas el paso",
    "utsa": "texas san antonio",
    "nc state": "north carolina state",
    "miami fl": "miami fl",
    "miami florida": "miami fl",
    "miami oh": "miami oh",
    "miami ohio": "miami oh",
    "st johns": "saint johns ny",
    "saint johns": "saint johns ny",
    "loyola chicago": "loyola il",
    "saint marys": "saint marys ca",
    "byu": "brigham young",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "fiu": "florida international",
    "fau": "florida atlantic",
    "etsu": "east tennessee state",
    "siu edwardsville": "southern illinois edwardsville",
    "cal": "california",
    "louisiana": "louisiana lafayette",
    "southern miss": "southern mississippi",
    "uc irvine": "california irvine",
    "unc asheville": "north carolina asheville",
    "unc greensboro": "north carolina greensboro",
    "unc wilmington": "north carolina wilmington",
    "umbc": "maryland baltimore county",
    "ut arlington": "texas arlington",
    "ut martin": "tennessee martin",
}


def slug_name(team_id: str) -> str:
    """School name from a Sports Reference id: .../schools/michigan-state/men -> 'michigan state'."""
    parts = str(team_id).strip("/").split("/")
    slug = parts[parts.index("schools") + 1] if "schools" in parts else parts[-1]
    return slug.replace("-", " ")


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation, spell out St. as state (trailing) or saint (leading)."""
    s = str(name).lower().replace("&", "").replace("'", "").replace("’", "")
    s = re.sub(r"[().,\-/]", " ", s)
    tokens = [t for t in s.split() if t not in ("the", "university", "univ")]
    if tokens and tokens[0] in ("st", "saint"):
        tokens[0] = "saint"
    if len(tokens) > 1 and tokens[-1] == "st":
        tokens[-1] = "state"
    return " ".join(tokens)


def match_names(names, team_table: pd.DataFrame, aliases: dict[str, str] | None = None) -> tuple[dict, list]:
    """Map each distinct name to a team id. `team_table` needs `team` (id) and `display_name`.

    Returns (mapping, unmatched). A name that matches two ids is reported as unmatched, never guessed.
    """
    aliases = ALIASES | (aliases or {})
    index: dict[str, set] = {}
    for team, display in zip(team_table["team"], team_table["display_name"]):
        # The id's school name ("michigan state") is the cleanest key; display names carry mascots.
        for key in {normalize_name(slug_name(team)), normalize_name(display)}:
            index.setdefault(key, set()).add(team)
    mapping, unmatched = {}, []
    for name in sorted({str(n) for n in names}):
        key = normalize_name(name)
        hits = index.get(key) or index.get(normalize_name(aliases.get(key, key))) or set()
        if len(hits) == 1:
            mapping[name] = next(iter(hits))
        else:
            unmatched.append(name)
    return mapping, unmatched


def map_team_columns(lines: pd.DataFrame, team_table: pd.DataFrame, cols=("team_a", "team_b"), aliases=None) -> tuple[pd.DataFrame, list]:
    """Replace names in `cols` with ids. Rows with an unmatched name are dropped; the unmatched names are returned."""
    names = pd.concat([lines[c] for c in cols])
    mapping, unmatched = match_names(names, team_table, aliases)
    out = lines.copy()
    for c in cols:
        out[c] = out[c].astype(str).map(mapping)
    return out.dropna(subset=list(cols)).reset_index(drop=True), unmatched
