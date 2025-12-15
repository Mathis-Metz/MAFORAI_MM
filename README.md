# Astronomical Transient Classification – ML & LLM Prototype

## Overview

This repository presents a prototype pipeline for astronomical transient classification
using SkyPortal-like JSON data. The objective is to demonstrate how heterogeneous,
deeply nested API outputs can be transformed into structured datasets suitable for
machine learning, and how Large Language Models (LLMs) can act as an astronomer
“copilot” during candidate vetting.

The focus is on design choices, robustness, and scientific realism rather than
model complexity or performance.

---

## Project Context

Modern time-domain surveys generate many transient candidates with incomplete
and noisy information. Platforms such as SkyPortal aggregate rich multi-modal data
(photometry, metadata, comments, follow-up requests, external catalogs), but this
data is not directly ML-ready.

The core challenge addressed here is structuring this data while preserving
uncertainty, and using LLMs to reason over partially available information.

---

## Data Structuring (Part 1)

Instead of flattening the full JSON schema, we extract a minimal, ML-relevant core
and store it in two Parquet datasets:

### Source-level dataset (`sources.parquet`)
One row per transient, including:
- sky position and detection score,
- SkyPortal-native flags (`is_transient`, `is_varstar`, `is_roid`),
- optional enrichment (redshift, TNS label),
- explicit availability flags (`has_tns`, `has_spectra`, `has_photometry`),
- simple counts of photometry and spectra.

### Lightcurve dataset (`lightcurves.parquet`)
Long-format photometry (one row per observation) with time, flux, filter, and
detection flag.

Missing data is propagated as `None` or empty structures and is never imputed.
The absence of information (e.g. no TNS report) is treated as informative.

---

## LLM-Based Astronomer Copilot (Part 2)

A local instruction-tuned LLM (Mistral) is used as a reasoning layer on top of the
structured data. The copilot:
- summarizes key information for each transient,
- assesses the likelihood of extragalactic origin,
- suggests appropriate follow-up observations,
- answers user-driven questions.

The prompt is stage-aware: TNS reports, spectra, and redshift measurements are
treated as high-confidence signals, while pipeline heuristic flags are treated as
lower-confidence.

---

## Evaluation & Limitations

This prototype is evaluated qualitatively. The copilot produces reasonable,
conservative summaries and follow-up suggestions, especially under limited
information.

No supervised ML models are trained, no image-based features are used, and no
external catalog retrieval is implemented. These choices are deliberate given
the scope of the exercise.

---

## Conclusion

This project demonstrates robust handling of complex astronomical JSON data,
principled feature selection, and a realistic integration of LLMs as scientific
copilots, with an emphasis on clarity and defensible design choices.
