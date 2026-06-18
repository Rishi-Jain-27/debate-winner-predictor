
import re
import pandas as pd

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
    
