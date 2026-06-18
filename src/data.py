"""
This file...
- loads three CSVs of data per each season with a season column
- joins Team_Info twice (`AffirmativeSchool`/`NegativeSchool` → `SchoolTeamCode`) for each school + seed-ELO
- parses the judges column to create a per-round judge list. Note that 895 rounds are panels.
- cleans SpeakerPoints.Points for both seasons — drops non-numeric points and out-of-range points (keep [0, 30])
- finds team-round speaks totals — groups SpeakerPoints by (RoundId, Side) and get a sum of the team's speaks.
- creates a per-debater round log of (debater, RoundId, season, side, outcome, Points, opponent, judges, stage) — good for sequential data input to the model
- parses RoundNumber to determine if is_prelim, elim (and what stage elim), and create a global round_order for sequencing.
"""

from pathlib import Path
import pandas as pd

# --- Constants --- #
DATA_DIR = Path(__file__).parent.parent / "data" # data/ lives a level above from src/
SEASONS = ["2024-25", "2025-26"]
ELIM_STAGES = {"D", "F", "O", "Q", "S"} # doubles, finals, octas, quarters, semis
ELIM_ORDER = {"D": 8, "O": 9, "Q": 10, "S": 11, "F": 12}

# --- Helper functions --- #

def _load_data(name: str, season: str) -> pd.DataFrame:
    # Read a CSV for a season and tag it with that season
    # File naming is {name}_{season}_HS.csv according to the existing data

    # pathing
    filename = f"{name}_{season}_HS.csv"
    path = DATA_DIR / filename

    # loading
    loaded = pd.read_csv(path)
    loaded["season"] = season # adds a column
    return loaded

def _clean_speaks(speaks: pd.DataFrame)  -> tuple[pd.DataFrame, int]:
    # drop the wrong/bad speakerpoints' rows
    # return the clean dataframe AND the number of rows dropped for the report to log

    speaker_points = pd.to_numeric(speaks["Points"], errors="coerce") # set non-numeric to NaN
    speaks = speaks.assign(Points=speaker_points) # write that back
    keep_mask = speaker_points.notna() & ((speaker_points >= 0) & (speaker_points <= 30)) # & is bitwise and
    clean_speaks = speaks[keep_mask] # masks on the original DF not on the Series (which is just speaker points)
    dropped_rows = (~keep_mask).sum() # ~ is bit-wise not

    return (clean_speaks, dropped_rows)

def _find_team_speaks_totals(clean_speaks: pd.DataFrame) -> pd.DataFrame:
    # team total = sum of 2 debaters per judge then avg across judges
    # collapses panels and duplicate name-variant rows
    per_judge = clean_speaks.groupby(["RoundId", "Side", "Judge"], as_index=False)["Points"].sum()
    totals = per_judge.groupby(["RoundId", "Side"], as_index=False)["Points"].mean()
    totals = totals.rename(columns={"Points": "team_speaks_total"})
    return totals


def _join_team_info(rounds: pd.DataFrame,
                    team_info: pd.DataFrame) -> pd.DataFrame:
    # attach school and elo to each round for both sides
    # remember team elo is end of season!

    aff_info = team_info.add_suffix("_aff")
    neg_info = team_info.add_suffix("_neg")

    out = (rounds.merge(aff_info, left_on="AffirmativeSchool", right_on="SchoolTeamCode_aff", how="left").merge(neg_info, left_on="NegativeSchool",   right_on="SchoolTeamCode_neg", how="left"))

    return out


def _get_round_num(rounds: pd.DataFrame) -> pd.DataFrame:
    # Get is_prelim, elim_stage, and the round_order from RoundNumber

    is_elim = rounds["RoundNumber"].isin(ELIM_STAGES)

    prelim_order = pd.to_numeric(rounds["RoundNumber"], errors="coerce")
    elim_order = rounds["RoundNumber"].map(ELIM_ORDER)
    round_order = prelim_order.fillna(elim_order).astype(int)

    return rounds.assign(
        is_prelim=(~is_elim),
        elim_stage=rounds["RoundNumber"].where(is_elim),
        round_order=round_order,
    )


# --- public functions --- #

def load_rounds() -> pd.DataFrame:
    # Loads each season's rounds and team info

    per_season = []

    for season in SEASONS:
        rounds = _load_data("Debate_Rounds", season)
        team_info = _load_data("Team_Info", season)

        df = _join_team_info(rounds, team_info)
        df = _get_round_num(df)

        per_season.append(df)
    
    rounds = pd.concat(per_season, ignore_index=True)
    rounds["aff_win"] = rounds["AffirmativeWin"].map({"Yes": 1, "No": 0})

    # Build totals
    speaks = pd.concat([_load_data("SpeakerPoints", season) for season in SEASONS], ignore_index=True)
    clean, _ = _clean_speaks(speaks)
    totals = _find_team_speaks_totals(clean)

    aff_total = totals[totals["Side"] == "Aff"].rename(columns={"team_speaks_total": "aff_speaks"})
    neg_total = totals[totals["Side"] == "Neg"].rename(columns={"team_speaks_total": "neg_speaks"})

    rounds = rounds.merge(aff_total[["RoundId", "aff_speaks"]], on="RoundId", how="left")
    rounds = rounds.merge(neg_total[["RoundId", "neg_speaks"]], on="RoundId", how="left")

    return rounds

def load_debater_log() -> pd.DataFrame:
    rounds = load_rounds()

    # Reshape rounds to be a per-side table
    aff = rounds.assign(
        side="Aff",
        team=rounds["AffirmativeSchool"],
        opponent=rounds["NegativeSchool"],
        won=rounds["aff_win"],
        own_team_speaks=rounds["aff_speaks"],
        opp_team_speaks=rounds["neg_speaks"],
    )

    neg = rounds.assign(
        side="Neg",
        team=rounds["NegativeSchool"],
        opponent=rounds["AffirmativeSchool"],
        won=1 - rounds["aff_win"], # flip
        own_team_speaks=rounds["neg_speaks"],
        opp_team_speaks=rounds["aff_speaks"],
    )
    cols = ["RoundId","season","side","team","opponent","won",
        "Judges","elim_stage","round_order","own_team_speaks","opp_team_speaks"]
    per_side = pd.concat([aff[cols], neg[cols]], ignore_index=True)


    # Get per-debater points from SpeakerPoints with groupby and mean
    speaks = pd.concat([_load_data("SpeakerPoints", s) for s in SEASONS], ignore_index=True)
    clean, _ = _clean_speaks(speaks)

    cols_for_speaks = ["RoundId","Side", "Debater"]
    debater_points = clean.groupby(cols_for_speaks, as_index=False)["Points"].mean()
    debater_points = debater_points.rename(columns={"Points": "own_points", "Side": "side"}) #type: ignore -- pylance is dumb

    # merge speaks data with round data
    log = debater_points.merge(per_side, on=["RoundId", "side"], how="left")
    return log

def report_cleaning(rounds, log, drop_counts) -> None:
    # print a report on dataset quality
    raise NotImplementedError # it's 1:00AM no way am I doing this

# Get the report
if __name__ == "__main__":
    rounds = load_rounds()
    debater_log = load_debater_log()
    # report_cleaning(rounds, debater_log, drop_counts)
