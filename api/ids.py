"""Team ids. The matrix uses Sports Reference ids; the API and the prediction tables use the school slug."""

from __future__ import annotations

import re

PREFIX = "www.sports-reference.com/cbb/schools/"
SUFFIX = "/men"
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,79}$")


def slug_of(team_id: str) -> str:
    """'www.sports-reference.com/cbb/schools/michigan-state/men' -> 'michigan-state'. Slugs pass through."""
    if team_id.startswith(PREFIX):
        return team_id[len(PREFIX):].removesuffix(SUFFIX).strip("/")
    return team_id


def team_id_of(slug: str) -> str:
    return f"{PREFIX}{slug}{SUFFIX}"


def is_slug(text: str) -> bool:
    return bool(SLUG_PATTERN.match(text))
