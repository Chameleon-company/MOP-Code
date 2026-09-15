# MOP Data Dashboard

This contribution builds a standalone data dashboard from the notebooks in
the public GitHub folder `master/usecases/FINALISED`.

## Architecture

1. Read FINALISED notebook files through the GitHub Contents API.
2. Extract explicit dataset references from notebook dataset sections and code.
3. Retrieve current metadata from the City of Melbourne Open Data API.
4. Read Use Case domains automatically from `usecases/Use_Case_Index.md`.
5. Normalise datasets and link each dataset to every use case that uses it.
6. Generate a static HTML dashboard and a CSV data register.

The current version has no server-side backend, database, framework or cloud
runtime. The generated HTML contains its data, styling and browser-side
interaction in one portable file.

## Capabilities

- Retrieval of the current FINALISED notebooks from the GitHub `master` branch
- City of Melbourne catalogue adoption and theme coverage
- Dataset-to-use-case traceability and reuse counts
- One combined Data Catalogue with dataset-scope, dataset-name, keyword, Use Case,
  source and theme filters, plus pagination
- Automatically refreshed Use Case Domain by City of Melbourne Theme matrix
- Stable asset identifiers backed by the asset ID registry
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
python3 data_dashboard.py
```

Alternatively, open and run `Refresh_Data_Dashboard.ipynb`.

Generated files:

- `outputs/MOP Data Dashboard.html`
- `outputs/Data_Catalogue.csv`

## Maintenance notes

- The dashboard is a generated snapshot. Run the builder again after FINALISED
  notebooks change.
- Dataset metadata that cannot be verified is shown as `Not stated`.
- Use Case domains are read automatically from `usecases/Use_Case_Index.md`.
  `config/use_case_domains.csv` is refreshed automatically as an offline cache.
- `config/asset_id_registry.csv` preserves stable identifiers between refreshes.
- GitHub Actions commits registry and domain-cache changes generated during a
  refresh, so newly assigned IDs remain stable in later runs.
- Website integration and any future refresh mechanism should follow the Web
  Development team's chosen architecture.
