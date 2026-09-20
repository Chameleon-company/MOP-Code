# MOP Data Dashboard

This contribution builds a standalone data dashboard from the notebooks in
the public GitHub folder `master/usecases/FINALISED`.

## Live dashboard

[Open the MOP Data Dashboard](https://chameleon-company.github.io/MOP-Code/)

The dashboard is designed to be deployed through GitHub Pages by
`.github/workflows/refresh-mop-data-dashboard.yml`. The link will become live
after this contribution is merged into `master`, GitHub Pages is configured to
use **GitHub Actions** as its publishing source, and the first deployment has
completed successfully.

## How to use the dashboard

1. Open the live dashboard using the link above.
2. Select **Data Usage Dashboard** to review headline statistics, MOP Open Data
   adoption, dataset reuse and the automatically calculated Use Case Domain by
   City of Melbourne Theme matrix.
3. Select **Data Catalogue** to browse the datasets associated with FINALISED
   Use Cases and the wider City of Melbourne Open Data catalogue.
4. Use the catalogue scope selector to view all datasets, MOP Open Data,
   datasets used by FINALISED Use Cases, unused MOP Open Data or external
   datasets.
5. Search by dataset name, description keyword, Use Case code, source,
   publisher, theme, data type, size or variable name. Additional filters and
   pagination can be used to narrow larger result sets.

If a search returns no matching dataset, the dashboard displays a no-results
message. Only Use Cases in `usecases/FINALISED` are included in the usage
analysis.

## Automatic refresh and deployment

The GitHub Actions workflow automatically rebuilds and republishes the
dashboard when relevant content is pushed to `master`. It also supports a
manual run from the repository's **Actions** page and a scheduled weekly
refresh.

During each refresh, the workflow:

1. reads the current FINALISED Use Case notebooks;
2. retrieves current City of Melbourne Open Data metadata;
3. recalculates the Use Case Domain by City of Melbourne Theme matrix from
   `usecases/Use_Case_Index.md`;
4. preserves existing stable dataset IDs and assigns IDs to newly discovered
   datasets;
5. regenerates `Data_Catalogue.csv` and `MOP Data Dashboard.html`; and
6. publishes the regenerated HTML file as the GitHub Pages website.

The refresh process therefore does not require the dashboard tables to be
edited manually. When new Use Cases are moved into `usecases/FINALISED` and the
changes reach `master`, the next matching workflow run incorporates them into
the published dashboard.

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
- Stable dataset identifiers backed by the stable ID registry
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

- `MOP Data Dashboard.html`
- `Data_Catalogue.csv`

## Maintenance notes

- The committed HTML and CSV files are generated snapshots. For local testing,
  run the builder again after FINALISED notebooks change. On `master`, the
  GitHub Actions workflow performs this refresh automatically.
- Dataset metadata that cannot be verified is shown as `Not stated`.
- Use Case domains are read automatically from `usecases/Use_Case_Index.md`.
  `use_case_domains.csv` is refreshed automatically as an offline cache.
- `asset_id_registry.csv` preserves stable dataset identifiers between refreshes.
- GitHub Actions commits registry and domain-cache changes generated during a
  refresh, so newly assigned IDs remain stable in later runs.
- The repository's GitHub Pages publishing source must be set to **GitHub
  Actions** for automatic deployment to the live URL.
