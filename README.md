# Transient Copilot

**Turning raw SkyPortal data into ML-ready datasets, with a local LLM as an astronomer's copilot.**

Time-domain surveys such as ZTF produce far more transient candidates than astronomers can inspect. Platforms like [SkyPortal](https://skyportal.io) gather everything known about each candidate (photometry, spectra, TNS reports, redshifts, comments), but as deeply nested JSON that no model can use directly.

This prototype does two things:

1. **Structures the data**: it extracts a minimal, ML-relevant core from SkyPortal-like JSON into two clean Parquet tables.
2. **Assists vetting**: a local LLM (Mistral 7B Instruct via [Ollama](https://ollama.com)) reads each structured candidate and writes a short assessment.

## 1. Data structuring

| dataset | one row per | content |
|---|---|---|
| `sources.parquet` | transient | position, detection score, SkyPortal flags (`is_transient`, `is_varstar`, `is_roid`), redshift, TNS label, availability flags (`has_tns`, `has_spectra`, `has_photometry`), photometry and spectra counts |
| `lightcurves.parquet` | photometric point | time (JD), measurement, error, filter, detection flag |

Design choices:

- **Keep only what a classifier needs** instead of flattening the whole schema.
- **Never impute missing data.** A missing value stays `None`, and its absence is recorded explicitly: a candidate with no TNS report is itself information.
- **Long format for light curves**, so that sources with any number of points and filters fit in one table.

## 2. LLM copilot

For each transient, the copilot builds a prompt from the structured row and a summary of its light curve, then asks the model to:

- summarize the key information,
- judge whether the source is likely extragalactic,
- recommend follow-up observations.

The prompt ranks evidence by reliability: TNS classifications, spectra and redshifts count as strong evidence, while pipeline flags count as weak heuristics. The model is asked to state its uncertainty.

Example output for `ZTF25aaktqzg` (abridged):

> **Key information:** high detection score (0.9999); reported to TNS and classified as SN Ia; redshift 0.0061; spectra available; observed for about 71 days in several filters.
>
> **Likely extragalactic:** yes, given the TNS classification and the redshift, although the pipeline's `is_transient` flag is false, which adds some uncertainty.

## Repository layout

```
src/
  parser.py        JSON loading, validation, field extraction
  dataset.py       source-level table
  lightcurves.py   long-format light-curve table
  llm_copilot.py   prompt building and Ollama calls
notebooks/
  Pipeline.ipynb   end-to-end run: inspection → Parquet → copilot
```

## Running it

```bash
pip install -r requirements.txt
ollama pull mistral:7b-instruct
```

Put a SkyPortal sources export (a JSON list of source objects) at `data/sources_sample.json`, then run `notebooks/Pipeline.ipynb`. The sample data used during development is not included in this repository.

## Scope and limitations

This is a prototype built in a few days, and its evaluation is qualitative. No supervised classifier is trained, no image features (cutouts) are used, and no external catalogs are queried. The natural next steps would be light-curve features (rise and decline rates, colors) and a classifier trained on TNS-labelled sources, with the LLM kept as an explanation layer.
