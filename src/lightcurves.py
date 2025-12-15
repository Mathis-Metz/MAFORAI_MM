import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser import parse_photometry


# =========================
# Conversion d'un point
# =========================


def photometry_point_to_row(
    source_id: str,
    point: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Convert a single photometry point into a flat row.

    Parameters
    ----------
    source_id : str
        Identifier of the source.
    point : Dict[str, Any]
        Photometry measurement dictionary.

    Returns
    -------
    Dict[str, Any] or None
        Flattened photometry row, or None if mandatory fields are missing.
    """
    if not isinstance(point, dict):
        return None

    jd = point.get("jd")
    flux = point.get("flux")

    if jd is None:
        return None

    filters = point.get("filters") or {}
    filt_name = filters.get("name") if isinstance(filters, dict) else None

    return {
        "id": source_id,
        "jd": jd,
        "flux": flux,
        "flux_err": point.get("fluxerr"),
        "filter": filt_name,
        "is_detection": flux is not None,
    }


# =========================
# Extraction d'une source
# =========================


def source_to_lightcurve_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all photometry points for a single source.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    List[Dict[str, Any]]
        List of photometry rows for the source.
    """
    source_id = obj.get("id")
    photometry = parse_photometry(obj)

    rows: List[Dict[str, Any]] = []

    for point in photometry:
        row = photometry_point_to_row(source_id, point)
        if row is not None:
            rows.append(row)

    return rows


# =========================
# Dataset global
# =========================


def build_lightcurve_dataset(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a long-format lightcurve dataset from JSON source objects.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of source objects parsed from JSON.

    Returns
    -------
    pandas.DataFrame
        Long-format photometry dataset.
    """
    rows = []

    for obj in data:
        if not isinstance(obj, dict):
            continue

        source_id = obj.get("id")
        if source_id is None:
            continue

        photometry = parse_photometry(obj)

        for point in photometry:
            row = photometry_point_to_row(source_id, point)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Sauvegarde
# =========================


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a lightcurve dataset to a Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Lightcurve dataset to save.
    path : str
        Output file path.
    """
    df.to_parquet(path, engine="pyarrow", index=False)
