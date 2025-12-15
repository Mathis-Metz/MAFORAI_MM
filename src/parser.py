import json
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Load the sources JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        List of source objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON object to be a list.")

    return data


def validate_source(obj: Dict[str, Any]) -> None:
    """
    Validate that a source object has the minimal required fields.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Raises
    ------
    KeyError
        If a required field is missing.
    """
    required_fields = ["id", "ra", "dec", "score"]

    for field in required_fields:
        if field not in obj:
            raise KeyError(f"Missing required field: {field}")


def parse_metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic metadata from a source object.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing basic source metadata.
    """
    validate_source(obj)

    return {
        "id": obj["id"],
        "ra": obj["ra"],
        "dec": obj["dec"],
        "redshift": obj.get("redshift"),
        "score": obj.get("score"),
        "is_transient": obj.get("transient"),
    }


def parse_label(obj: Dict[str, Any]) -> str | None:
    """
    Extract the astrophysical class label from TNS information.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    str or None
        Name of the astrophysical class if available, otherwise None.
    """
    tns = obj.get("tns_info") or {}
    obj_type = tns.get("object_type")

    if obj_type is None:
        return None

    return obj_type.get("name")


def parse_photometry(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the raw photometry list from a source object.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    List[Dict[str, Any]]
        List of photometry measurement dictionaries.
    """
    tns = obj.get("tns_info") or {}
    photometry = tns.get("photometry") or []

    if not isinstance(photometry, list):
        return []

    return photometry


def parse_spectra(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the spectra metadata from a source object.

    Parameters
    ----------
    obj : Dict[str, Any]
        Source object parsed from JSON.

    Returns
    -------
    List[Dict[str, Any]]
        List of spectra metadata dictionaries.
    """
    tns = obj.get("tns_info") or {}
    spectra = tns.get("spectra") or []

    if not isinstance(spectra, list):
        return []

    return spectra
