"""
This builds the non sequential inputs
- judge features: rounds_seen, aff_winrate, leniency, seen flag -- all per round and averaged over the panel
- static team features: debaters_rounds_seen, is_new flag, school_prior_win_rate, confidence_weight -- also perround

shrinkage to reduce noise:
value = (n*observed + k*prior)/(n+k); confidence w = n/(n+k).
We walk rounds in an imposed chronological order
"""

import collections
import pandas as pd

# --- Constants --- #
SHRINK_K = 5.0  # prior weight for shrinkage
AFF_WIN_PRIOR = 0.5 # global prior for an aff win
LENIENCY_SCALE = 10.0 # scale for centered judge leniency

# --- Helper functions --- #

def _shrink(n: int, observed: float, prior: float) -> float:
    # confidence-weighted shrinkage toward a prior
    return (n * observed + SHRINK_K * prior) / (n + SHRINK_K)

def _confidence(n: int) -> float:
    # confidence weight w = n / (n + k)
    return n / (n + SHRINK_K)

def _parse_judges(judges) -> list:
    # "A, B, C" -> ["A", "B", "C"]; NaN/blank -> []
    if pd.isna(judges) or not str(judges).strip():
        return []
    return [j.strip() for j in str(judges).split(",") if j.strip()]

# --- public functions --- #

def build_judge_features(rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Per-round panel-averaged judge features

    Args:
        rounds: columns RoundId, global_order, Judges, aff_win, round_speaks.

    Returns:
        DataFrame: RoundId, judge_rounds_seen, judge_aff_winrate, judge_leniency, judge_seen.
    """
    global_speaks = rounds["round_speaks"].mean(skipna=True)

    n_seen = collections.defaultdict(int)
    aff_wins = collections.defaultdict(float)
    win_n = collections.defaultdict(int)
    speaks_sum = collections.defaultdict(float)
    speaks_n = collections.defaultdict(int)

    out_rows = []
    for r in rounds.sort_values("global_order").itertuples(index=False):
        judges = _parse_judges(r.Judges)

        # panel-average each judge's as-of-before stats
        seens, winrates, leniencies = [], [], []
        for j in judges:
            n = n_seen[j]
            seens.append(n)
            wr = (aff_wins[j] / win_n[j]) if win_n[j] else AFF_WIN_PRIOR
            winrates.append(_shrink(win_n[j], wr, AFF_WIN_PRIOR))
            lev = (speaks_sum[j] / speaks_n[j]) if speaks_n[j] else global_speaks
            leniencies.append(_shrink(speaks_n[j], lev, global_speaks))

        if judges:
            rounds_seen = sum(seens) / len(seens)
            aff_winrate = sum(winrates) / len(winrates)
            leniency = (sum(leniencies) / len(leniencies) - global_speaks) / LENIENCY_SCALE
            seen_flag = 1.0 if max(seens) > 0 else 0.0
        else:
            rounds_seen, aff_winrate, leniency, seen_flag = 0.0, AFF_WIN_PRIOR, 0.0, 0.0

        out_rows.append((r.RoundId, rounds_seen, aff_winrate, leniency, seen_flag))

        # update AFTER recording
        for j in judges:
            n_seen[j] += 1
            if pd.notna(r.aff_win):
                aff_wins[j] += r.aff_win # type: ignore
                win_n[j] += 1
            if pd.notna(r.round_speaks):
                speaks_sum[j] += r.round_speaks # type: ignore
                speaks_n[j] += 1

    return pd.DataFrame(out_rows, columns=["RoundId", "judge_rounds_seen", "judge_aff_winrate", "judge_leniency", "judge_seen"])

def build_static_features(membership: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(round, side) static team features

    Args:
        membership: one row per (RoundId, side, debater_id) with columns
            RoundId, side, debater_id, won, school_id, global_order.

    Returns:
        DataFrame: RoundId, side, deb_seen_mean, deb_seen_min, side_is_new,
                   school_winrate, school_conf.
    """
    # gather each round-side's members (+ school, outcome), keyed for ordered walk
    rounds: dict = {}
    for row in membership.itertuples(index=False):
        info = rounds.setdefault((row.RoundId, row.side),
                                 {"order": row.global_order, "won": row.won, "school": row.school_id, "debaters": []})
        info["debaters"].append(row.debater_id)

    deb_seen = collections.defaultdict(int)
    school_wins = collections.defaultdict(float)
    school_n = collections.defaultdict(int)

    out_rows = []
    for key in sorted(rounds, key=lambda k: rounds[k]["order"]):
        rid, side = key
        info = rounds[key]
        debaters, school, won = info["debaters"], info["school"], info["won"]

        seens = [deb_seen[d] for d in debaters] or [0]
        deb_seen_mean = sum(seens) / len(seens)
        deb_seen_min = min(seens)
        side_is_new = 1.0 if max(seens) == 0 else 0.0

        n = school_n[school]
        obs = (school_wins[school] / n) if n else AFF_WIN_PRIOR
        school_winrate = _shrink(n, obs, AFF_WIN_PRIOR)
        school_conf = _confidence(n)

        out_rows.append((rid, side, deb_seen_mean, deb_seen_min, side_is_new, school_winrate, school_conf))

        # update AFTER recording
        for d in debaters:
            deb_seen[d] += 1
        if pd.notna(won):
            school_wins[school] += won
            school_n[school] += 1

    return pd.DataFrame(out_rows, columns=["RoundId", "side", "deb_seen_mean", "deb_seen_min", "side_is_new", "school_winrate", "school_conf"])
