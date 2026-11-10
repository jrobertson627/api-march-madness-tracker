"""Which groups of features move a prediction, by ablation.

For a matchup, set one group's differences to zero and see how far the win probability for team A moves. With every
group set to zero the model returns exactly 50% (it is symmetric), so effects are measured from a fair coin. Effects
do not add up exactly because features interact; the leftover is reported as `interaction`.

These describe what the model does with its inputs. They are not causes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.dataset import is_signed

GROUPS = {
    "efficiency": ["diff_adj_em", "diff_adj_off", "diff_adj_def", "diff_off_eff", "diff_def_eff", "diff_eff_margin", "proj_margin"],
    "tempo": ["diff_adj_tempo", "diff_tempo"],
    "four_factors": ["diff_efg", "diff_tov_pct", "diff_orb_pct", "diff_ftr", "diff_def_efg", "diff_def_tov_pct", "diff_def_orb_pct",
                     "diff_def_ftr", "diff_fg3_rate", "diff_fg3_pct", "diff_ft_pct"],
    "form": ["diff_last5_margin", "diff_last10_margin", "diff_form_trend", "diff_win_pct", "diff_avg_margin"],
    "schedule": ["diff_sos_elo", "diff_sos_adj"],
    "elo": ["diff_elo"],
    "roster": ["diff_returning_min_pct", "diff_incoming_prior_min", "diff_returning_bpm_wavg", "diff_top5_min_share",
               "diff_top_player_min_share", "diff_top_player_bpm", "diff_depth"],
    "coach": ["diff_coach_seasons_at_school", "diff_coach_prior_ncaa_apps"],
    "seed": ["diff_seed"],
}
# One headline number per group, shown next to each team in an explanation.
HEADLINE = {
    "efficiency": "adj_em", "tempo": "adj_tempo", "four_factors": "efg", "form": "last10_margin", "schedule": "sos_elo",
    "elo": "elo", "roster": "returning_min_pct", "coach": "coach_seasons_at_school", "seed": "seed",
}


def group_columns(columns) -> dict[str, list[str]]:
    """Map each group to the model input columns it owns. Signed columns in no group go to 'other'."""
    cols = set(columns)
    out = {g: [c for c in members if c in cols] for g, members in GROUPS.items()}
    owned = {c for members in out.values() for c in members}
    other = sorted(c for c in cols if is_signed(c) and c not in owned and c != "loc_a")
    if other:
        out["other"] = other
    return {g: c for g, c in out.items() if c}


def _win_probability(predictor, rows: pd.DataFrame) -> np.ndarray:
    """The model's calibrated probability for team A, without computing scores."""
    f = predictor.fitted
    p = f.model.predict_proba(rows[f.cols])
    return f.calibrator.transform(p) if f.calibrator is not None else p


def factor_effects(predictor, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per input row: p_a_win, an effect per group (p_a_win minus the probability with the group zeroed),
    and `interaction` (what the groups do not explain of p_a_win - 0.5)."""
    used = group_columns(predictor.fitted.cols)
    p_full = _win_probability(predictor, rows)
    out = pd.DataFrame({"p_a_win": p_full}, index=rows.index)
    total = np.zeros(len(rows))
    for group, cols in used.items():
        ablated = rows.copy()
        ablated[cols] = 0.0
        out[group] = p_full - _win_probability(predictor, ablated)
        total += out[group].to_numpy()
    out["interaction"] = (p_full - 0.5) - total
    return out


def check_complete(columns) -> list[str]:
    """Signed model inputs that no group owns (an empty list means every input is explained by some group)."""
    return group_columns(columns).get("other", [])
