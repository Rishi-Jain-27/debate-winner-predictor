# this file could use some better docstrings ngl and better organization

import re
import pandas as pd
from data import _load_data
import collections

# --- Key algorithms --- #
# This is the same thing as creating the debater id and returning it
# bc debater id == that debater's canonical key
def create_canonical_key(name) -> str:
    """
    Creates the canonical key for a name.

    Args:
        name: str of "{first_name} {last_name}" - including middle name if present.
        Can also be NaN, None, "", or a string of whitespaces

    Returns:
        canoncial_key (str): either "" for invalid inputs OR the canonical key of the name
    """

    # Guard against empty/just whitespace or NaN/None input
    if pd.isna(name) or (name.strip() == ""):
        return ""
    
    # Create the canonical key
    canonical_key = name.lower()
    canonical_key = re.sub(r"[^\w\s]", r"", canonical_key)
    canonical_key = canonical_key.split() # gives each part of the name as a list of strings
    canonical_key = sorted(canonical_key) # then sorts them alphabetically
    canonical_key = " ".join(canonical_key)

    return canonical_key

# this is canonical key but no sorting -- instead, return last name first initial
# this is just an experimental function
def key_with_abbrev(name) -> str:
    # Guard against empty/just whitespace or NaN/None input
    if pd.isna(name) or (name.strip() == ""):
        return ""
    
    # Create the key
    key = name.lower()
    key = re.sub(r"[^\w\s]", r"", key)
    key = key.split()

    if len(key) == 1:
        return key[0]
    elif len(key) == 0:
        return ""
    else:
        return (key[-1] + " " + key[0][0]) # last name first initial

# Abbreviation aware key helper
def _parse_name(name):
    """
    Parses a name into its identity components, assuming first-last order
    (as in Team_Info's FullName columns).

    Args:
        name: str of "{first} [middle ...] {last}", or NaN/None/""/whitespace.

    Returns:
        (surname, first_token, is_initial) on success:
            surname (str): the last token in lowercase.
            first_token (str): the first token in lowercased (full name or a lone initial).
            is_initial (bool): True if first_token is a single character (e.g. "o").
        "" if the name is missing/blank.

    Middle tokens are ignored (only first and last are used).
    Single-token names (e.g. "Madonna") return that token as BOTH surname and first_token, with is_initial False.
    """
    
    if pd.isna(name) or (name.strip() == ""):
        return ""
    lower = name.lower()
    lower = re.sub(r"[^\w\s]", r"", lower)
    tokens = lower.split()
    return (tokens[-1], tokens[0], len(tokens[0]) == 1)

def _build_res_map(names) -> dict:
    """
    Builds the abbreviation-resolution map.

    For every name with a FULL first name, collect it under (surname, first_initial)
    Keep only the combos that have one full first name (ambiguous ones are dropped).

    Args:
        names: an iterable of raw name strings.
        Missing/blank names are skipped.

    Returns:
        dict mapping (surname, initial) -> full_first_name.
        Only unambiguous combos are included -- there is only ONE value per key.
    """
     
    ambig_res_map = collections.defaultdict(set)
    for name in names:
        if _parse_name(name) == "":
            continue
        last, first, is_init = _parse_name(name)
        if not is_init:
            ambig_res_map[(last, first[0])].add(first)
    
    res_map = {}
    for key, val_set in ambig_res_map.items():
        if len(val_set) == 1:
            res_map[key] = list(val_set)[0]
    
    return res_map

def smart_abbrev_keying(name, res_map):
    if _parse_name(name) == "":
        return ""
    last, first, is_init = _parse_name(name)

    # if it is a single letter, look it up in rez map and expand it if found
    # if not found, fallback to last + " " + first
    if not is_init:
        return (last + " " + first)
    else:
        if (last, first) in res_map:
            return (last + " " + res_map[(last, first)])
        return (last + " " + first)

def derive_school_id(school_name) -> str:
    """
    Finds the school_id given a school name.

    Args:
        school_name: str, NaN, or None.
    
    Returns:
        school_id: str of the school's id.
    """

    # Guard against empty/just whitespace or NaN/None input
    if pd.isna(school_name) or (school_name.strip() == ""):
        return ""

    school_id = school_name.lower()
    school_id = school_id.split()
    school_id = " ".join(school_id)
    return school_id

# --- Attach keys algorithm(s) --- #
def add_ids() -> pd.DataFrame:
    """
    Adds debater_id1, debater_id2, and school_id to Team_Info

    Returns:
        pd.DataFrame: contans SchoolTeamCode, debater_id1, debater_id2,
        and school_id in team_info across both seasons
    """
    # Load seasons
    team_info_24 = _load_data("Team_Info", "2024-25")
    team_info_25 = _load_data("Team_Info", "2025-26")

    pooled_names = pd.concat([team_info_24["FullName1"], team_info_24["FullName2"], team_info_25["FullName1"], team_info_25["FullName2"]])
    res_map = _build_res_map(pooled_names)

    # Concat them
    team_info = pd.concat([team_info_25, team_info_24], ignore_index=True)

    # Apply create_canonical_key
    team_info["debater_id1"] = team_info["FullName1"].apply(lambda n: smart_abbrev_keying(n, res_map))
    team_info["debater_id2"] = team_info["FullName2"].apply(lambda n: smart_abbrev_keying(n, res_map))

    # Derive school id
    team_info["school_id"] = team_info["SchoolName"].apply(derive_school_id)

    # drop dups -- dups are if they match on ["SchoolTeamCode", "debater_id1", "debater_id2", "school_id"]
    team_info = team_info.drop_duplicates(subset=["SchoolTeamCode", "debater_id1", "debater_id2", "school_id"])

    return team_info[["SchoolTeamCode","debater_id1","debater_id2","school_id"]]

# --- Diagnostic algorithms --- #
def get_cross_season_match_rate() -> float:
    """
    Gets the cross-season match rate.

    Returns:
        match_rate (float): the cross-season match rate
        representing what fraction of 24-25 debaters come back in 25-26
    """
    # Load seasons
    team_info_24 = _load_data("Team_Info", "2024-25")
    team_info_25 = _load_data("Team_Info", "2025-26")

    pooled_names = pd.concat([team_info_24["FullName1"], team_info_24["FullName2"], team_info_25["FullName1"], team_info_25["FullName2"]])
    res_map = _build_res_map(pooled_names)

    ids_2024 = set(pd.concat([team_info_24["FullName1"], team_info_24["FullName2"]], ignore_index=True).apply(lambda n: smart_abbrev_keying(n, res_map))) - {""}
    ids_2025 = set(pd.concat([team_info_25["FullName1"], team_info_25["FullName2"]], ignore_index=True).apply(lambda n: smart_abbrev_keying(n, res_map))) - {""}

    # running diagnostics show that smart_abbrev_keying is MUCH better
    # ids_2024 = set(pd.concat([team_info_24["FullName1"], team_info_24["FullName2"]], ignore_index=True).apply(key_with_abbrev)) - {""}
    # ids_2025 = set(pd.concat([team_info_25["FullName1"], team_info_25["FullName2"]], ignore_index=True).apply(key_with_abbrev)) - {""}

    # ids_2024 = set(pd.concat([team_info_24["FullName1"], team_info_24["FullName2"]], ignore_index=True).apply(create_canonical_key)) - {""}
    # ids_2025 = set(pd.concat([team_info_25["FullName1"], team_info_25["FullName2"]], ignore_index=True).apply(create_canonical_key)) - {""}

    intersection = ids_2024 & ids_2025

    match_rate = len(intersection) / len(ids_2024) # of the 24-25 debater, what fraction reappear in 25-26

    return match_rate

def find_collision_count(season: str) -> float:
    # Load seasons
    team_info = _load_data("Team_Info", season)
    unique_canonical = len(set(pd.concat([team_info["FullName1"], team_info["FullName2"]], ignore_index=True).apply(create_canonical_key)) - {""})
    unique_abbrev = len(set(pd.concat([team_info["FullName1"], team_info["FullName2"]], ignore_index=True).apply(key_with_abbrev)) - {""})
    collision_count = unique_canonical - unique_abbrev
    return (collision_count) / unique_canonical