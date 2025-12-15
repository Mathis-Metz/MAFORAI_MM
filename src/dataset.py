import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.parser import (
    validate_source,
    parse_metadata,
    parse_label,
    parse_photometry,
    parse_spectra,
)


def source_to_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single source JSON object into a flat dictionary.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    Dict[str, Any]
        Flattened representation of the source.
    """
    if not isinstance(obj, dict):
        return None

    validate_source(obj)

    photometry = parse_photometry(obj)
    spectra = parse_spectra(obj)

    return {
        # Core identity
        "id": obj["id"],
        "ra": obj["ra"],
        "dec": obj["dec"],
        "score": obj.get("score"),
        # SkyPortal-native flags
        "is_transient": obj.get("transient"),
        "is_varstar": obj.get("varstar"),
        "is_roid": obj.get("is_roid"),
        # Enrichment (optional)
        "redshift": obj.get("redshift"),
        "label": parse_label(obj),
        # Availability flags
        "has_tns": isinstance(obj.get("tns_info"), dict),
        "has_redshift": obj.get("redshift") is not None,
        "has_photometry": len(photometry) > 0,
        "has_spectra": len(spectra) > 0,
        # Counts
        "n_photometry": len(photometry),
        "n_spectra": len(spectra),
    }


def build_dataset(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a source-level dataset from a list of JSON source objects.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of source objects parsed from JSON.

    Returns
    -------
    pandas.DataFrame
        Source-level dataset.
    """
    rows = []

    for obj in data:
        if not isinstance(obj, dict):
            continue

        row = source_to_row(obj)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to a Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to save.
    path : str
        Output file path.
    """
    df.to_parquet(path, engine="pyarrow", index=False)
