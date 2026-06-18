
import re
import pandas as pd
from data import _load_data

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

def add_ids() -> pd.DataFrame:
    """
    Adds debater_id1, debater_id2, and school_id to Team_Info

    Returns:
        pd.DataFrame: contans SchoolTeamCode, debater_id1, debater_id2,
        and school_id in team_info across both seasons
    """
    # Load seasons
    team_info_1 = _load_data("Team_Info", "2024-25")
    team_info_2 = _load_data("Team_Info", "2025-26")

    # Concat them
    team_info = pd.concat([team_info_2, team_info_1], ignore_index=True)

    # Apply create_canonical_key
    team_info["debater_id1"] = team_info["FullName1"].apply(create_canonical_key)
    team_info["debater_id2"] = team_info["FullName2"].apply(create_canonical_key)

    # Derive school id
    team_info["school_id"] = team_info["SchoolName"].apply(derive_school_id)

    # drop dups -- dups are if they match on ["SchoolTeamCode", "debater_id1", "debater_id2", "school_id"]
    team_info = team_info.drop_duplicates(subset=["SchoolTeamCode", "debater_id1", "debater_id2", "school_id"])

    return team_info[["SchoolTeamCode","debater_id1","debater_id2","school_id"]]

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

    ids_2024 = set(pd.concat([team_info_24["FullName1"], team_info_24["FullName2"]], ignore_index=True).apply(key_with_abbrev)) - {""}
    ids_2025 = set(pd.concat([team_info_25["FullName1"], team_info_25["FullName2"]], ignore_index=True).apply(key_with_abbrev)) - {""}

    intersection = ids_2024 & ids_2025

    match_rate = len(intersection) / len(ids_2024) # of the 24-25 debater, what fraction reappear in 25-26

    return match_rate
