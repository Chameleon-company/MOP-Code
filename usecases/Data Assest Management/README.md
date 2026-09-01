# MOP Data Asset Management Platform

This contribution builds a standalone data-asset dashboard from the notebooks in
the public GitHub folder `master/usecases/FINALISED`.

## Architecture

1. Read FINALISED notebook files through the GitHub Contents API.
2. Extract explicit dataset references from notebook dataset sections and code.
3. Retrieve current metadata from the City of Melbourne Open Data API.
4. Normalise datasets and link each asset to every use case that uses it.
5. Generate a static HTML dashboard and a CSV data register.

The current version has no server-side backend, database, framework or cloud
runtime. The generated HTML contains its data, styling and browser-side
interaction in one portable file.

## Capabilities

- Retrieval of the current FINALISED notebooks from the GitHub `master` branch
- City of Melbourne catalogue adoption and theme coverage
- Dataset-to-use-case traceability and reuse counts
- Search, source and theme filters, and pagination
- Stable asset identifiers and optional manual metadata overrides
- Standalone HTML and CSV outputs suitable for website integration

## Runtime resources

- Python 3.10 or later
- Jupyter only when running the notebook interface
- Network access to the public GitHub API, raw GitHub content and the City of
  Melbourne Open Data API
- Optional `GITHUB_TOKEN` environment variable when a higher GitHub API rate
  limit is required

No third-party Python package is required by the builder.

## Run

From this directory:

```bash
python3 data_asset_catalogue.py
```

Alternatively, open and run `Refresh_Data_Asset_Catalogue.ipynb`.

Generated files:

- `outputs/Data_Asset_Catalogue.html`
- `outputs/Data_Asset_Catalogue.csv`

## Maintenance notes

- The dashboard is a generated snapshot. Run the builder again after FINALISED
  notebooks change.
- Dataset metadata that cannot be verified is shown as `Not stated`.
- `config/asset_overrides.csv` is reserved for evidence-based corrections that
  cannot be obtained from the source APIs.
- `config/use_case_domains.csv` maintains the confirmed domain labels used by
  the existing dashboard matrix.
- Website integration and any future refresh mechanism should follow the Web
  Development team's chosen architecture.
