"""The tournament bracket as slots and rounds.

A bracket is a list of slots in bracket order. Neighbouring slots meet, then neighbouring winners, and so on, so the
tree is implied by the order. A slot holds one team, or two when they play in (the First Four). For the real
tournament the 64 slots are four regions of 16, in the order 1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15,
with the regions ordered so that the two semifinals pair neighbouring regions.

Rounds: 1 is the round of 64 ... 6 is the championship. The First Four is round 0 in the Phase 1 data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOT_SEEDS = (1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15)
REACH_COLUMNS = ("p_round_of_64", "p_round_of_32", "p_sweet_16", "p_elite_8", "p_final_four", "p_championship_game", "p_champion")


@dataclass(frozen=True)
class Bracket:
    teams: tuple          # team ids; the position is the row and column in a probability matrix
    seed: tuple           # seed per team (None for a generic bracket)
    region: tuple         # region per team (None for a generic bracket)
    slots: tuple          # per slot: tuple of 1 or 2 team positions
    regions: tuple = ()   # regions in bracket order (empty for a generic bracket)

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def n_rounds(self) -> int:
        return int(np.log2(self.n_slots))

    @property
    def index(self) -> dict:
        return {t: i for i, t in enumerate(self.teams)}

    @property
    def playin_slots(self) -> list:
        return [k for k, s in enumerate(self.slots) if len(s) == 2]

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_slots(cls, teams, slots, seed=None, region=None, regions=()) -> "Bracket":
        """A generic bracket: slots in bracket order, each a tuple of 1 or 2 team positions."""
        n = len(slots)
        if n < 2 or n & (n - 1):
            raise ValueError(f"the number of slots must be a power of two, got {n}")
        used = [t for s in slots for t in s]
        if sorted(used) != list(range(len(teams))) or any(len(s) not in (1, 2) for s in slots):
            raise ValueError("every team must appear in exactly one slot, and a slot holds one or two teams")
        return cls(tuple(teams), tuple(seed) if seed is not None else (None,) * len(teams),
                   tuple(region) if region is not None else (None,) * len(teams), tuple(tuple(s) for s in slots), tuple(regions))

    @classmethod
    def from_teams(cls, teams: pd.DataFrame, final_four=None) -> "Bracket":
        """From a table with team, seed and region. `final_four` is ((region, region), (region, region)): the two
        semifinals. If omitted the regions are taken in sorted order, which is only a guess for a real tournament."""
        missing = {"team", "seed", "region"} - set(teams.columns)
        if missing:
            raise ValueError(f"bracket table is missing columns {sorted(missing)}")
        t = teams.drop_duplicates("team").reset_index(drop=True)
        t["seed"] = t["seed"].astype(int)
        regions = sorted(t["region"].unique())
        if len(regions) != 4:
            raise ValueError(f"a bracket needs four regions, found {regions}")
        if final_four is None:
            final_four = ((regions[0], regions[1]), (regions[2], regions[3]))
        order = tuple(r for pair in final_four for r in pair)
        if sorted(order) != regions:
            raise ValueError(f"final_four {final_four} must name each region exactly once ({regions})")
        index = {team: i for i, team in enumerate(t["team"])}
        slots = []
        for region in order:
            g = t[t["region"] == region]
            counts = g["seed"].value_counts()
            bad = [s for s in range(1, 17) if counts.get(s, 0) not in (1, 2)]
            if bad:
                raise ValueError(f"region {region}: seeds {bad} do not have one or two teams")
            if len(g) not in range(16, 19):
                raise ValueError(f"region {region} has {len(g)} teams")
            for s in SLOT_SEEDS:
                slots.append(tuple(index[x] for x in g[g["seed"] == s]["team"]))
        return cls.from_slots(list(t["team"]), slots, seed=list(t["seed"]), region=list(t["region"]), regions=order)

    @classmethod
    def from_results(cls, teams: pd.DataFrame, results: pd.DataFrame) -> "Bracket":
        """For a past season: add First Four teams that have no seed row (they take their opponent's slot) and read
        the Final Four pairing from the semifinal results. `results` needs team_a, team_b and round."""
        t = teams[["team", "seed", "region"]].copy()
        known = set(t["team"])
        for r in results[results["round"] == 0].itertuples():
            for me, other in ((r.team_a, r.team_b), (r.team_b, r.team_a)):
                if me not in known and other in known:
                    row = t[t["team"] == other].iloc[0]
                    t.loc[len(t)] = [me, row["seed"], row["region"]]
                    known.add(me)
        region_of = dict(zip(t["team"], t["region"]))
        semis = results[results["round"] == 5]
        pairs = [tuple(region_of[x] for x in (r.team_a, r.team_b)) for r in semis.itertuples()]
        final_four = tuple(pairs) if len(pairs) == 2 and len({r for p in pairs for r in p}) == 4 else None
        return cls.from_teams(t, final_four)

    # ------------------------------------------------------------------ replay against real results

    def replay(self, results: pd.DataFrame):
        """Feed real results through the structure.

        `results` needs team_a, team_b, winner (team ids) and round. Returns (actual, problems): `actual[l]` is an
        array of the winner position per game at level l (-1 when unknown; level 0 is the slots after play-ins), and
        `problems` lists anything that does not fit: a game between teams the structure says cannot meet, a game with
        no result, or a result in the wrong round. An empty list means the structure matches reality.
        """
        idx = self.index
        done = {}
        problems = []
        for r in results.itertuples():
            if r.team_a not in idx or r.team_b not in idx:
                problems.append(f"result involves a team that is not in the bracket: {r.team_a} v {r.team_b}")
                continue
            a, b = idx[r.team_a], idx[r.team_b]
            done[frozenset((a, b))] = (idx[r.winner], int(r.round))
        used = set()

        def play(a, b, rnd):
            key = frozenset((a, b))
            if key not in done:
                problems.append(f"no result for round {rnd}: {self.teams[a]} v {self.teams[b]}")
                return -1
            w, rr = done[key]
            used.add(key)
            if rr != rnd:
                problems.append(f"round mismatch for {self.teams[a]} v {self.teams[b]}: data says {rr}, structure says {rnd}")
            return w

        level0 = np.empty(self.n_slots, dtype=int)
        for k, cand in enumerate(self.slots):
            level0[k] = cand[0] if len(cand) == 1 else play(cand[0], cand[1], 0)
        actual = [level0]
        for rnd in range(1, self.n_rounds + 1):
            prev = actual[-1]
            nxt = np.empty(len(prev) // 2, dtype=int)
            for j in range(len(nxt)):
                a, b = prev[2 * j], prev[2 * j + 1]
                nxt[j] = play(a, b, rnd) if a >= 0 and b >= 0 else -1
            actual.append(nxt)
        for key in done:
            if key not in used:
                a, b = sorted(key)
                problems.append(f"result does not fit the bracket: {self.teams[a]} v {self.teams[b]}")
        return actual, problems

    def realized_reach(self, actual) -> np.ndarray:
        """(teams, rounds + 1) 0/1 array: reached round 1 ... reached round n, then won the title. From `replay`."""
        out = np.zeros((len(self.teams), self.n_rounds + 1))
        for rnd in range(0, self.n_rounds + 1):
            for t in actual[rnd]:
                if t >= 0:
                    out[t, rnd] = 1.0  # the winner at level rnd is a participant in round rnd + 1 (or the champion)
        return out
