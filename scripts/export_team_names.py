"""Write artifacts/models/team_names.csv: team id and display name, for matching betting lines.

    python scripts/export_team_names.py

Read-only query. Use ml.team_names.match_names to map a lines file's names to these ids and see what is unmatched.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data.db import get_pool, query  # noqa: E402

OUT = ROOT / "artifacts" / "models" / "team_names.csv"


def main() -> None:
    rows = query(
        """
        SELECT s.canonical_source_path AS team, s.display_name,
               min(ss.ending_year) AS first_season, max(ss.ending_year) AS last_season
        FROM schools s
        JOIN school_seasons ss ON ss.school_id = s.id
        GROUP BY s.canonical_source_path, s.display_name
        ORDER BY s.display_name
        """
    )
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} teams to {OUT}")
    get_pool().close()


if __name__ == "__main__":
    main()
