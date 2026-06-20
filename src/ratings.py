"""
This file creates
- previous elo per debater by walking rounds chronologically
- gives each debater a base elo
- rating for a round == rating before it
- no dates in data so tournament order is approximate
- builds an ONLINE (as-of-before) Elo rating per debater by walking rounds in
"""

import collections
import pandas as pd

# --- Constants --- #
BASE_ELO = 1500.0
K_FACTOR = 24.0

# --- Helper functions --- #

def _expected(rating_a: float, rating_b: float) -> float:
    # expected score (win prob) of A vs B under the Elo logistic
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

# --- public functions --- #

def compute_online_elo(membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk rounds in global order and produce as-of-before Elo.

    Args:
        membership: one row per (RoundId, side, debater_id) with columns
            RoundId, side, debater_id, global_order, aff_win

    Returns:
        debater_pre: each debater's rating BEFORE that round.
        team_pre: each team's mean rating BEFORE that round.
    """
    # collect each round's members + outcome, keyed by RoundId
    rounds: dict = {}
    for row in membership.itertuples(index=False):
        info = rounds.setdefault(row.RoundId, {"order": row.global_order, "aff_win": row.aff_win, "Aff": [], "Neg": []})
        info[row.side].append(row.debater_id)

    elo: dict = collections.defaultdict(lambda: BASE_ELO)
    deb_rows = []
    team_rows = []

    # process in imposed chronological order
    for rid in sorted(rounds, key=lambda r: rounds[r]["order"]):
        info = rounds[rid]
        aff, neg = info["Aff"], info["Neg"]

        team_aff = sum(elo[d] for d in aff) / len(aff) if aff else BASE_ELO
        team_neg = sum(elo[d] for d in neg) / len(neg) if neg else BASE_ELO

        # record pre-round ratings (BEFORE any update for this round)
        for d in aff:
            deb_rows.append((rid, d, elo[d]))
        for d in neg:
            deb_rows.append((rid, d, elo[d]))
        team_rows.append((rid, "Aff", team_aff))
        team_rows.append((rid, "Neg", team_neg))

        # update only when the label is known and both sides are present
        aff_win = info["aff_win"]
        if aff and neg and pd.notna(aff_win):
            delta = K_FACTOR * (aff_win - _expected(team_aff, team_neg))
            for d in aff:
                elo[d] += delta
            for d in neg:
                elo[d] -= delta

    debater_pre = pd.DataFrame(deb_rows, columns=["RoundId", "debater_id", "pre_elo"])
    team_pre = pd.DataFrame(team_rows, columns=["RoundId", "side", "team_pre_elo"])
    return debater_pre, team_pre
