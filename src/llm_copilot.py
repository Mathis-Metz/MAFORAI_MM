from typing import Any, Dict, Tuple

import pandas as pd
import ollama


# =========================
# Configuration du LLM
# =========================

MODEL_NAME = "mistral:7b-instruct"

SYSTEM_PROMPT = (
    "You are an expert astronomy assistant helping astronomers classify "
    "optical transients and decide whether follow-up observations are needed.\n"
    "Important:\n"
    "- Some sources are early-stage candidates with limited information.\n"
    "- Some sources are already reported to TNS and may have confirmed classifications.\n"
    "- TNS reports, spectra, and redshift measurements are high-confidence signals.\n"
    "- Pipeline heuristic flags (e.g. is_transient) are lower-confidence.\n"
    "Be concise and explicit about uncertainty."
)

# =========================
# Chargement des datasets
# =========================


def load_datasets(
    sources_path: str,
    lightcurves_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load structured Parquet datasets produced by the data processing pipeline.

    Parameters
    ----------
    sources_path : str
        Path to the source-level Parquet file.
    lightcurves_path : str
        Path to the lightcurve Parquet file.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        Source-level dataset and lightcurve dataset.
    """
    sources_df = pd.read_parquet(sources_path)
    lightcurves_df = pd.read_parquet(lightcurves_path)

    return sources_df, lightcurves_df


# =========================
# Résumé des lightcurves
# =========================


def summarize_lightcurve(lc: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a simple summary of a lightcurve for downstream reasoning.

    The summary is intentionally lightweight and interpretable, avoiding
    complex feature engineering.

    Parameters
    ----------
    lc : pandas.DataFrame
        Lightcurve data for a single source.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing basic statistics describing the lightcurve.
    """
    detections = lc[lc["flux"].notna()]

    summary: Dict[str, Any] = {
        "n_points": int(len(lc)),
        "n_detections": int(len(detections)),
        "filters": sorted(lc["filter"].dropna().unique().tolist()),
    }

    if not detections.empty:
        summary.update(
            {
                "flux_min": float(detections["flux"].min()),
                "flux_max": float(detections["flux"].max()),
                "time_span_days": float(
                    detections["jd"].max() - detections["jd"].min()
                ),
            }
        )
    else:
        summary.update(
            {
                "flux_min": None,
                "flux_max": None,
                "time_span_days": None,
            }
        )

    return summary


# =========================
# Construction du prompt
# =========================


def build_prompt(
    source_row: Dict[str, Any],
    lc_summary: Dict[str, Any],
) -> str:
    """
    Build a structured prompt for the LLM-based astronomer copilot.

    Parameters
    ----------
    source_row : Dict[str, Any]
        Dictionary containing source-level metadata.
    lc_summary : Dict[str, Any]
        Summary statistics of the lightcurve.

    Returns
    -------
    str
        Prompt string to be sent to the LLM.
    """
    prompt = f"""
The astronomer is mainly interested in:
- Supernovae
- Kilonovae
- Other extragalactic transients

They want to avoid:
- Galactic variable stars
- Minor objects
- Artifacts

=== Source metadata ===
ID: {source_row.get("id")}
RA / Dec: {source_row.get("ra")}, {source_row.get("dec")}
Detection score: {source_row.get("score")}

Pipeline flags:
- Is transient (heuristic): {source_row.get("is_transient")}
- Is variable star: {source_row.get("is_varstar")}
- Is minor object: {source_row.get("is_roid")}

High-confidence signals:
- TNS reported: {source_row.get("has_tns")}
- Redshift available: {source_row.get("has_redshift")}
- Spectra available: {source_row.get("has_spectra")}
- Redshift value: {source_row.get("redshift")}
- TNS label (if any): {source_row.get("label")}

=== Lightcurve summary ===
Number of photometry points: {lc_summary.get("n_points")}
Number of detections: {lc_summary.get("n_detections")}
Filters used: {lc_summary.get("filters")}
Flux range: {lc_summary.get("flux_min")} to {lc_summary.get("flux_max")}
Observed time span (days): {lc_summary.get("time_span_days")}

=== Task ===
1. Summarize the key information an astronomer should know.
2. Is this transient likely extragalactic? Briefly justify.
3. Does this transient deserve follow-up observations?
4. If yes, suggest what kind of follow-up (e.g. spectroscopy, more photometry).

Be concise and explicit about uncertainty.
"""
    return prompt.strip()


# =========================
# Interaction avec le LLM
# =========================


def query_llm(prompt: str) -> str:
    """
    Query the local Mistral LLM via Ollama.

    Parameters
    ----------
    prompt : str
        Prompt to send to the language model.

    Returns
    -------
    str
        Textual response generated by the LLM.
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.3,
            "num_ctx": 4096,
        },
    )

    return response["message"]["content"]


# =========================
# Pipeline principal
# =========================


def run_copilot_for_source(
    source_id: str,
    sources: pd.DataFrame,
    lightcurves: pd.DataFrame,
) -> str:
    """
    Run the LLM-based copilot pipeline for a single transient.

    Parameters
    ----------
    source_id : str
        Identifier of the transient to analyze.
    sources : pandas.DataFrame
        Source-level dataset.
    lightcurves : pandas.DataFrame
        Lightcurve dataset.

    Returns
    -------
    str
        Copilot-generated summary and follow-up suggestion.
    """
    source_row = sources[sources["id"] == source_id].iloc[0].to_dict()

    lc = lightcurves[lightcurves["id"] == source_id]

    lc_summary = summarize_lightcurve(lc)
    prompt = build_prompt(source_row, lc_summary)

    return query_llm(prompt)
