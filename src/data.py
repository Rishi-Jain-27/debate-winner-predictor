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

import pandas as pd

def load_rounds():
    pass

def load_debater_log():
    pass

def report_cleaning():
    pass

def _load_csv():
    pass

def _join_teaminfo():
    pass

def _clean_pts():
    pass

def _find_team_spks_totals():
    pass

def _parse_round_num():
    pass

if __name__ == '__main__':
    pass

