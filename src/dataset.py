"""
This file takes the cleaned data (see data.py, not perfect but good enough) and creates the Dataset
it:
- adds a debater_id onto the log
This file (Phase 3) turns the cleaned data into a leak-safe PyTorch Dataset:
- uses a global round key bc round_id repeats across seasons
- creates a chronological order, our dataset has no dates so this is approximate
- attaches previous elo
- builds a list of each debater's earlier rounds
- finds yields per round as the aff_seq, neg_seq and masks, static, and
judge features (features.py), the win label and speaker total targets
"""

import re
import collections

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import _load_data, load_rounds, load_debater_log, SEASONS
from names import _build_res_map, smart_abbrev_keying, add_ids
import ratings
import features

# --- Constants --- #
MAX_HISTORY = 30 # how much history each debater has
TOKEN_DIM = 7 # bc there is 6 token features + a recency feature
STATIC_DIM = 10 # there are 5 static features per side, and 2 sides
JUDGE_DIM = 4 # judge features

POINTS_SCALE = 30.0
ELO_SCALE = 200.0
SPEAKS_SCALE = 60.0 # speaks is total among partners
RECENCY_SCALE = 50.0
SEEN_SCALE = 5.0 # divides ln(1 + rounds seen)

# stored token features — recency is added per-query in _history
TOKEN_STORE_COLS = ["tok_points", "tok_won", "tok_side_aff", "tok_is_elim", "tok_pre_elo", "tok_opp_elo"]

# --- Helper functions --- #

def _res_map() -> dict:
    # abbreviation res map, built from both seasons' Team_Info names (first-last)
    pool = []
    for s in SEASONS:
        ti = _load_data("Team_Info", s)
        pool += [ti["FullName1"], ti["FullName2"]]
    return _build_res_map(pd.concat(pool))

def _resolve_debater(name, res_map: dict) -> str:
    # SpeakerPoints is last-first, reverse tokens to first-last, then reuse the resolver
    if pd.isna(name) or not re.search(r"[A-Za-z]", str(name)):
        return ""
    first_last = " ".join(reversed(str(name).split()))
    return smart_abbrev_keying(first_last, res_map)

def _add_global_order(rounds: pd.DataFrame) -> pd.DataFrame:
    # impose a total order: season -> tournament -> round_order
    # exact within a tournament, approx for all the tournaments order
    keys = rounds[["round_uid", "season", "Tournament", "round_order"]].drop_duplicates()
    keys = keys.sort_values(["season", "Tournament", "round_order"]).reset_index(drop=True)
    keys["global_order"] = range(len(keys))
    return keys[["round_uid", "global_order"]]

def _build_log(rounds: pd.DataFrame, res_map: dict) -> pd.DataFrame:
    # er-debater log: stable ids, global order, as-of-before Elo, token features
    log = load_debater_log()
    log = log.reset_index(drop=True)
    log["round_uid"] = log["season"] + "|" + log["RoundId"]

    # stable debater_id, unresolved names get a placeholder
    log["debater_id"] = log["Debater"].apply(lambda n: _resolve_debater(n, res_map))
    blank = log.index[log["debater_id"] == ""]
    log.loc[blank, "debater_id"] = ["__unk__" + str(i) for i in blank]

    # round-level context
    order = _add_global_order(rounds)
    log = log.merge(order, on="round_uid", how="left")
    log = log.merge(rounds[["round_uid", "Tournament", "aff_win"]], on="round_uid", how="left")
    log["is_elim"] = log["elim_stage"].notna().astype(float)

    # school_id per side via the team to members map
    team_school = add_ids()
    code_to_school = dict(zip(team_school["SchoolTeamCode"], team_school["school_id"]))
    log["school_id"] = log["team"].map(code_to_school).fillna("")

    # one row per (round, side, debater), collapses name-variant duplicates
    log = log.drop_duplicates(subset=["round_uid", "side", "debater_id"])

    # previous elo
    mem = log[["round_uid", "side", "debater_id", "global_order", "aff_win"]].rename(columns={"round_uid": "RoundId"})
    debater_pre, team_pre = ratings.compute_online_elo(mem)
    debater_pre = debater_pre.rename(columns={"RoundId": "round_uid"})
    team_pre = team_pre.rename(columns={"RoundId": "round_uid"})
    log = log.merge(debater_pre, on=["round_uid", "debater_id"], how="left")
    opp = team_pre.rename(columns={"side": "opp_side", "team_pre_elo": "opp_team_pre_elo"})
    log["opp_side"] = log["side"].map({"Aff": "Neg", "Neg": "Aff"})
    log = log.merge(opp, on=["round_uid", "opp_side"], how="left")

    # token features
    log["tok_points"] = (log["own_points"] / POINTS_SCALE).fillna(0.0)
    log["tok_won"] = log["won"].fillna(0.5)
    log["tok_side_aff"] = (log["side"] == "Aff").astype(float)
    log["tok_is_elim"] = log["is_elim"].fillna(0.0)
    log["tok_pre_elo"] = ((log["pre_elo"] - 1500.0) / ELO_SCALE).fillna(0.0)
    log["tok_opp_elo"] = ((log["opp_team_pre_elo"] - 1500.0) / ELO_SCALE).fillna(0.0)

    return log

def _static_vec(row) -> list:
    # one side's 5 static features (scaled)
    if row is None:
        return [0.0, 0.0, 1.0, 0.5, 0.0]  # fully-novel side: is_new=1, school prior=0.5, conf=0
    return [
        np.log1p(row["deb_seen_mean"]) / SEEN_SCALE,
        np.log1p(row["deb_seen_min"]) / SEEN_SCALE,
        float(row["side_is_new"]),
        float(row["school_winrate"]),
        float(row["school_conf"]),
    ]

def _judge_vec(row) -> list:
    # panel-averaged judge features (scaled)
    if row is None:
        return [0.0, 0.5, 0.0, 0.0]
    return [
        np.log1p(row["judge_rounds_seen"]) / SEEN_SCALE,
        float(row["judge_aff_winrate"]),
        float(row["judge_leniency"]),
        float(row["judge_seen"]),
    ]

# --- Dataset --- #
class DebateDataset(Dataset):
    """
    __getitem__ returns a dict of tensors:
    - aff_seq/neg_seq , (2, MAX_HISTORY, TOKEN_DIM), two debaters' history token sequences
    - aff_mask/neg_mask, (2, MAX_HISTORY), 1 is real token, 0 is padding
    - static, (static_dim), aff and neg static features
    - judge, (judge_dim), panel averaged judge features
    - label (scalar) aff win (1 or 0)
    - speaker_targets (2), aff or neg team speak totals / speaks scale
    - speaker_mask (2), 1 is target present (so no forfeits)
    """

    def __init__(self):
        rounds = load_rounds()
        rounds["round_uid"] = rounds["season"] + "|" + rounds["RoundId"]
        rounds["round_speaks"] = rounds[["aff_speaks", "neg_speaks"]].mean(axis=1)
        rounds = rounds.merge(_add_global_order(rounds), on="round_uid", how="left")

        log = _build_log(rounds, _res_map())

        # per-debater ordered token arrays
        log = log.sort_values(["debater_id", "global_order"])
        self.deb_tokens = {}
        for did, grp in log.groupby("debater_id", sort=False):
            self.deb_tokens[did] = (
                grp["global_order"].to_numpy(),
                grp[TOKEN_STORE_COLS].to_numpy(dtype=float),
            )

        # round_uid == {"Aff": [ids], "Neg": [ids]}
        members = collections.defaultdict(lambda: {"Aff": [], "Neg": []})
        for r in log.itertuples(index=False):
            members[r.round_uid][r.side].append(r.debater_id) # type: ignore

        # Feature tables
        static = features.build_static_features(
            log[["round_uid", "side", "debater_id", "won", "school_id", "global_order"]].rename(columns={"round_uid": "RoundId"})
        ).rename(columns={"RoundId": "round_uid"})
        judge = features.build_judge_features(
            rounds[["round_uid", "global_order", "Judges", "aff_win", "round_speaks"]].rename(columns={"round_uid": "RoundId"})
        ).rename(columns={"RoundId": "round_uid"})
        static_aff = static[static["side"] == "Aff"].set_index("round_uid")
        static_neg = static[static["side"] == "Neg"].set_index("round_uid")
        judge_idx = judge.set_index("round_uid")

        # one example per labeled round
        self.examples = []
        for r in rounds.itertuples(index=False):
            if pd.isna(r.aff_win):
                continue
            uid = r.round_uid
            saff = static_aff.loc[uid] if uid in static_aff.index else None
            sneg = static_neg.loc[uid] if uid in static_neg.index else None
            jrow = judge_idx.loc[uid] if uid in judge_idx.index else None

            aff_sp, neg_sp = r.aff_speaks, r.neg_speaks
            self.examples.append({
                "aff_ids": members[uid]["Aff"],
                "neg_ids": members[uid]["Neg"],
                "order": int(r.global_order), # type: ignore
                "static": np.array(_static_vec(saff) + _static_vec(sneg), dtype=np.float32),
                "judge": np.array(_judge_vec(jrow), dtype=np.float32),
                "label": float(r.aff_win), # type: ignore
                "speaks": np.array([
                    (aff_sp / SPEAKS_SCALE) if pd.notna(aff_sp) else 0.0, # type: ignore
                    (neg_sp / SPEAKS_SCALE) if pd.notna(neg_sp) else 0.0, # type: ignore
                ], dtype=np.float32),
                "speaks_mask": np.array([float(pd.notna(aff_sp)), float(pd.notna(neg_sp))], dtype=np.float32),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def _history(self, debater_id: str, order: int):
        # this debater's tokens with global_order < order, last max history and padded at the front
        out = np.zeros((MAX_HISTORY, TOKEN_DIM), dtype=np.float32)
        mask = np.zeros(MAX_HISTORY, dtype=np.float32)
        if debater_id not in self.deb_tokens:
            return out, mask
        orders, toks = self.deb_tokens[debater_id]
        cut = int(np.searchsorted(orders, order, side="left"))   # strictly-earlier cutoff (excludes own round)
        start = max(0, cut - MAX_HISTORY)
        sel = toks[start:cut]
        if len(sel) == 0:
            return out, mask
        recency = ((order - orders[start:cut]) / RECENCY_SCALE).reshape(-1, 1)
        feat = np.concatenate([sel, recency], axis=1)            # [m, TOKEN_DIM]
        m = feat.shape[0]
        out[MAX_HISTORY - m:] = feat
        mask[MAX_HISTORY - m:] = 1.0
        return out, mask

    def _team(self, ids: list, order: int):
        # up to 2 debaters == (2, MAX_HISTORY, TOKEN_DIM) + (2, MAX_HISTORY), a missing slot is empty
        seqs = np.zeros((2, MAX_HISTORY, TOKEN_DIM), dtype=np.float32)
        masks = np.zeros((2, MAX_HISTORY), dtype=np.float32)
        for i, did in enumerate(ids[:2]):
            seqs[i], masks[i] = self._history(did, order)
        return seqs, masks

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        aff_seq, aff_mask = self._team(ex["aff_ids"], ex["order"])
        neg_seq, neg_mask = self._team(ex["neg_ids"], ex["order"])
        return {
            "aff_seq": torch.from_numpy(aff_seq),
            "aff_mask": torch.from_numpy(aff_mask),
            "neg_seq": torch.from_numpy(neg_seq),
            "neg_mask": torch.from_numpy(neg_mask),
            "static": torch.from_numpy(ex["static"]),
            "judge": torch.from_numpy(ex["judge"]),
            "label": torch.tensor(ex["label"], dtype=torch.float32),
            "speaker_targets": torch.from_numpy(ex["speaks"]),
            "speaker_mask": torch.from_numpy(ex["speaks_mask"]),
        }


if __name__ == "__main__":
    ds = DebateDataset()
    print("examples:", len(ds))
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k:16} {tuple(v.shape)}")
