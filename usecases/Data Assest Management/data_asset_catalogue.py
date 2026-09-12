"""Build the MOP Data Asset Management Platform from GitHub FINALISED use cases.

The builder uses only the Python standard library. It reads public notebooks from
GitHub, retrieves the City of Melbourne Open Data catalogue, normalises dataset
references, and writes a standalone HTML dashboard plus a CSV register.
"""

from __future__ import annotations

import ast
import csv
import html
import json
import os
import posixpath
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
GITHUB_OWNER = "Chameleon-company"
GITHUB_REPOSITORY = "MOP-Code"
GITHUB_BRANCH = "master"
FINALISED_PATH = "usecases/FINALISED"
CITY_CATALOGUE_API = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
CITY_DATA_HOSTS = {"data.melbourne.vic.gov.au", "www.data.melbourne.vic.gov.au"}
DATA_FILE_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".geojson", ".xlsx", ".xls", ".zip",
    ".parquet", ".feather", ".pickle", ".pkl", ".shp", ".kml",
}
GENERIC_LABELS = {"catalog", "catalogue", "data", "dataset", "datasets", "standards"}
NON_DATA_HOSTS = {
    "cdnjs.cloudflare.com", "cdn.jsdelivr.net", "code.jquery.com",
    "fonts.googleapis.com", "pandas.pydata.org", "scikit-learn.org",
    "plot.ly", "plotly.com", "youtube.com", "www.youtube.com",
}

URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+", re.I)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", re.I)
HTML_LINK_RE = re.compile(
    r"<a\s+[^>]*href=[\"'](https?://[^\"']+)[\"'][^>]*>(.*?)</a>",
    re.I | re.S,
)
READ_RE = re.compile(
    r"(?:pd\.)?read_(?:csv|table|excel|json|parquet|feather|pickle)"
    r"\s*\(\s*[fr]?[\"']([^\"']+)",
    re.I,
)
MOP_CALL_RE = re.compile(
    r"\b(?:collect_data|api_unlimited)\s*\(\s*[fr]?[\"']([a-z0-9][a-z0-9-]*)[\"']",
    re.I,
)
USE_CASE_RE = re.compile(r"UC\d{5}(?!\d)", re.I)
SECTION_HEADING_RE = re.compile(r"(?:^|\n)\s*#{1,6}\s+", re.I)
HTML_SECTION_RE = re.compile(
    r"class=[\"'][^\"']*(?:usecase-(?:unnumbered|section|sub-section|sub-sub-section)-heading|usecase-contents)",
    re.I,
)


@dataclass(frozen=True)
class BuildConfig:
    project_dir: Path
    github_owner: str = GITHUB_OWNER
    github_repository: str = GITHUB_REPOSITORY
    github_branch: str = GITHUB_BRANCH
    finalised_path: str = FINALISED_PATH

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "outputs"

    @property
    def overrides_file(self) -> Path:
        return self.project_dir / "config" / "asset_overrides.csv"

    @property
    def use_case_domains_file(self) -> Path:
        return self.project_dir / "config" / "use_case_domains.csv"

    @property
    def asset_ids_file(self) -> Path:
        return self.project_dir / "config" / "asset_id_registry.csv"


def _request(url: str, *, expect_json: bool = False, retries: int = 2) -> Any:
    headers = {
        "Accept": "application/vnd.github+json" if "api.github.com" in url else "application/json",
        "User-Agent": "MOP-data-asset-platform",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                payload = response.read()
            return json.loads(payload) if expect_json else payload.decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == retries:
                raise RuntimeError(f"Unable to retrieve {url}: {error}") from error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to retrieve {url}")


def _github_contents(config: BuildConfig, path: str) -> list[dict[str, Any]]:
    encoded_path = quote(path, safe="/")
    query = urlencode({"ref": config.github_branch})
    url = (
        f"{GITHUB_API}/repos/{config.github_owner}/{config.github_repository}"
        f"/contents/{encoded_path}?{query}"
    )
    payload = _request(url, expect_json=True)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a GitHub directory at {path}")
    return payload


def list_finalised_notebooks(config: BuildConfig) -> list[dict[str, str]]:
    """Return notebook metadata from the remote GitHub FINALISED directory."""
    notebooks: list[dict[str, str]] = []
    pending = [config.finalised_path]
    while pending:
        directory = pending.pop()
        for item in _github_contents(config, directory):
            if item.get("type") == "dir":
                pending.append(item["path"])
            elif item.get("type") == "file" and item.get("name", "").lower().endswith(".ipynb"):
                notebooks.append(
                    {
                        "path": item["path"],
                        "download_url": item["download_url"],
                        "html_url": item["html_url"],
                    }
                )
    return sorted(notebooks, key=lambda item: item["path"])


def fetch_city_catalogue() -> dict[str, dict[str, Any]]:
    """Retrieve the complete City of Melbourne catalogue with safe pagination."""
    results: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urlencode({"limit": 100, "offset": offset})
        payload = _request(f"{CITY_CATALOGUE_API}?{query}", expect_json=True)
        batch = payload.get("results", [])
        total = int(payload.get("total_count", 0))
        results.extend(batch)
        if not batch:
            break
        offset += len(batch)
    return {
        item["dataset_id"].lower(): item
        for item in results
        if item.get("dataset_id")
    }


def _cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _clean(value: Any, limit: int = 500) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _normalise_url(url: str) -> str:
    cleaned = html.unescape(url).rstrip(".,;，；")
    parsed = urlparse(cleaned)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", parsed.query, ""))


def _mop_dataset_id(reference: str) -> str | None:
    parsed = urlparse(reference)
    if parsed.netloc.lower() not in CITY_DATA_HOSTS:
        return None
    match = re.search(r"/(?:explore/dataset|catalog/datasets)/([^/?#]+)", parsed.path, re.I)
    if match:
        return match.group(1).lower()
    query = parse_qs(parsed.query)
    if query.get("dataset"):
        return query["dataset"][0].lower()
    return None


def _repo_path_from_url(reference: str, config: BuildConfig) -> str | None:
    parsed = urlparse(reference)
    parts = [part for part in parsed.path.split("/") if part]
    owner, repository = config.github_owner.lower(), config.github_repository.lower()
    if parsed.netloc.lower() == "raw.githubusercontent.com" and len(parts) >= 4:
        if parts[0].lower() == owner and parts[1].lower() == repository:
            return "/".join(parts[3:])
    if parsed.netloc.lower() in {"github.com", "www.github.com"} and len(parts) >= 5:
        if parts[0].lower() == owner and parts[1].lower() == repository and parts[2] in {"blob", "raw"}:
            return "/".join(parts[4:])
    return None


def _looks_like_data_url(reference: str) -> bool:
    parsed = urlparse(reference)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host in NON_DATA_HOSTS or not host:
        return False
    if host in CITY_DATA_HOSTS:
        return _mop_dataset_id(reference) is not None
    if host == "raw.githubusercontent.com" or PurePosixPath(path).suffix in DATA_FILE_EXTENSIONS:
        return True
    return any(marker in path for marker in ("/api/", "/dataset/", "/datasets/", "/resource/", "/download/"))


def _dataset_section_cells(cells: list[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    inside = False
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "markdown":
            continue
        text = _cell_text(cell)
        plain = _clean(text).lower()
        if not inside and re.fullmatch(r"data\s*sets?", plain):
            inside = True
            yield index, cell
            continue
        if inside and (SECTION_HEADING_RE.search(text) or HTML_SECTION_RE.search(text)):
            break
        if inside:
            yield index, cell


def _display_name(reference: str) -> str:
    parsed = urlparse(reference)
    name = PurePosixPath(parsed.path).name or parsed.netloc
    return _clean(re.sub(r"[-_]+", " ", name).title()) or "Not stated"


def _static_dataset_ids(code: str) -> set[str]:
    """Resolve literal dataset IDs passed to the two FINALISED API helpers."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    strings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    strings[target.id] = node.value.value

    def value_of(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return strings.get(node.id)
        return None

    dataset_ids: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        argument_index = 0 if name == "fetch_melbourne_dataset" else 1 if name == "fetch_data" else -1
        if argument_index < 0 or len(node.args) <= argument_index:
            continue
        value = value_of(node.args[argument_index])
        if value and re.fullmatch(r"[a-z0-9][a-z0-9-]+", value, re.I):
            dataset_ids.add(value.lower())
    return dataset_ids


def _source_details(reference: str) -> tuple[str, str]:
    host = urlparse(reference).netloc.lower()
    if host in CITY_DATA_HOSTS:
        return "City of Melbourne Open Data", "City of Melbourne"
    if host.endswith("data.vic.gov.au") or host.endswith("vic.gov.au"):
        return "External government open data — Victorian Government", "Victorian Government"
    if host.endswith("abs.gov.au"):
        return "External government open data — Australian Bureau of Statistics", "Australian Bureau of Statistics"
    if host:
        return f"External data source — {host}", host
    return "Repository or local file", "Not stated"


def _reference_record(
    reference: str,
    label: str,
    notebook: dict[str, str],
    config: BuildConfig,
) -> dict[str, str]:
    reference = _normalise_url(reference) if reference.startswith(("http://", "https://")) else reference
    mop_id = _mop_dataset_id(reference) if reference.startswith("http") else None
    repo_path = _repo_path_from_url(reference, config) if reference.startswith("http") else None

    if not reference.startswith("http"):
        base = posixpath.dirname(notebook["path"])
        repo_path = posixpath.normpath(posixpath.join(base, reference))
        encoded = quote(repo_path, safe="/")
        reference = (
            f"https://raw.githubusercontent.com/{config.github_owner}/{config.github_repository}/"
            f"{config.github_branch}/{encoded}"
        )

    if mop_id:
        key = f"MOP:{mop_id}"
    elif repo_path:
        key = f"REPO:{repo_path.lower()}"
    else:
        key = f"URL:{reference.lower()}"

    source, publisher = _source_details(reference)
    if repo_path:
        source, publisher = "Repository or local file", config.github_repository

    return {
        "asset_key": key,
        "dataset": (_display_name(reference) if _clean(label).lower() in GENERIC_LABELS else _clean(label)) or _display_name(reference),
        "link": reference,
        "source": source,
        "publisher": publisher,
        "notebook_code": (USE_CASE_RE.search(notebook["path"]).group(0).upper() if USE_CASE_RE.search(notebook["path"]) else PurePosixPath(notebook["path"]).stem),
        "notebook_url": notebook["html_url"],
    }


def scan_notebook(notebook: dict[str, str], config: BuildConfig) -> list[dict[str, str]]:
    """Extract explicit dataset references from one remote notebook."""
    document = json.loads(_request(notebook["download_url"]))
    cells = document.get("cells", [])
    candidates: dict[str, str] = {}

    for _, cell in _dataset_section_cells(cells):
        text = _cell_text(cell)
        for label, url in MARKDOWN_LINK_RE.findall(text):
            candidates[_normalise_url(url)] = _clean(label)
        for url, label in HTML_LINK_RE.findall(text):
            candidates[_normalise_url(url)] = _clean(label)

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        text = _cell_text(cell)
        for dataset_id in MOP_CALL_RE.findall(text):
            url = f"{CITY_CATALOGUE_API}/{dataset_id.lower()}"
            candidates.setdefault(url, dataset_id.replace("-", " ").title())
        for dataset_id in _static_dataset_ids(text):
            url = f"{CITY_CATALOGUE_API}/{dataset_id}"
            candidates.setdefault(url, dataset_id.replace("-", " ").title())
        for reference in READ_RE.findall(text):
            if reference.startswith(("http://", "https://")):
                if _looks_like_data_url(reference):
                    candidates.setdefault(_normalise_url(reference), "")
            else:
                candidates.setdefault(reference, "")
        for url in URL_RE.findall(text):
            url = _normalise_url(url)
            if _looks_like_data_url(url):
                candidates.setdefault(url, "")

    return [
        _reference_record(reference, label, notebook, config)
        for reference, label in sorted(candidates.items())
    ]


def _type_from_link(link: str) -> str:
    extension = PurePosixPath(urlparse(link).path).suffix.lower()
    return {
        ".csv": "Tabular (CSV)", ".tsv": "Tabular (TSV)",
        ".xlsx": "Spreadsheet (XLSX)", ".xls": "Spreadsheet (XLS)",
        ".json": "JSON", ".geojson": "Geospatial (GeoJSON)",
        ".parquet": "Tabular (Parquet)", ".zip": "Archive (ZIP)",
        ".shp": "Geospatial vector", ".kml": "Geospatial (KML)",
    }.get(extension, "Not stated")


def _load_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["asset_key"].strip(): {key: value.strip() for key, value in row.items() if value and key != "asset_key"}
            for row in csv.DictReader(handle)
            if row.get("asset_key", "").strip()
        }


def _load_asset_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["asset_key"].strip(): row["asset_id"].strip()
            for row in csv.DictReader(handle)
            if row.get("asset_key", "").strip() and row.get("asset_id", "").strip()
        }


def _legacy_asset_key(record: dict[str, Any]) -> str:
    if record["asset_key"].startswith("MOP:"):
        return record["asset_key"]
    parsed = urlparse(record["link"])
    if parsed.netloc:
        return f"{parsed.netloc}{parsed.path}".lower().rstrip("/")
    return PurePosixPath(parsed.path).name.lower()


def _assign_asset_ids(records: list[dict[str, Any]], path: Path) -> None:
    registry = _load_asset_ids(path)
    highest = max(
        (
            int(match.group(1))
            for value in registry.values()
            if (match := re.fullmatch(r"ASSET-(\d+)", value))
        ),
        default=0,
    )
    for record in sorted(records, key=lambda item: item["asset_key"]):
        key = _legacy_asset_key(record)
        asset_id = registry.get(record["asset_key"]) or registry.get(key)
        if not asset_id:
            highest += 1
            asset_id = f"ASSET-{highest:04d}"
            registry[key] = asset_id
        record["asset_id"] = asset_id

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_key", "asset_id"])
        writer.writeheader()
        writer.writerows(
            {"asset_key": key, "asset_id": value}
            for key, value in sorted(registry.items())
        )


def build_records(
    findings: list[dict[str, str]],
    city_catalogue: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for finding in findings:
        grouped[finding["asset_key"]].append(finding)

    records: list[dict[str, Any]] = []
    for dataset_id, item in city_catalogue.items():
        key = f"MOP:{dataset_id}"
        meta = item.get("metas", {}).get("default", {})
        fields = item.get("fields", []) or []
        field_names = [field.get("name", "") for field in fields if field.get("name")]
        field_types = {field.get("type", "") for field in fields}
        uses = grouped.pop(key, [])
        record = {
            "asset_id": "",
            "asset_key": key,
            "dataset": _clean(meta.get("title")) or dataset_id.replace("-", " ").title(),
            "link": f"https://data.melbourne.vic.gov.au/explore/dataset/{dataset_id}/",
            "source": "City of Melbourne Open Data",
            "publisher": _clean(meta.get("publisher")) or "City of Melbourne",
            "themes": meta.get("theme") or ["Not stated"],
            "used_by": sorted({(use["notebook_code"], use["notebook_url"]) for use in uses}),
            "datatype": "Geospatial" if field_types & {"geo_point_2d", "geo_shape"} else "Tabular",
            "size": str(meta.get("records_count")) if meta.get("records_count") is not None else "Not stated",
            "variables": field_names,
            "kind": "mop",
        }
        records.append(record)

    for key, uses in grouped.items():
        first = uses[0]
        record = {
            "asset_id": "",
            "asset_key": key,
            "dataset": first["dataset"],
            "link": first["link"],
            "source": first["source"],
            "publisher": first["publisher"],
            "themes": ["Not applicable"],
            "used_by": sorted({(use["notebook_code"], use["notebook_url"]) for use in uses}),
            "datatype": _type_from_link(first["link"]),
            "size": "Not stated",
            "variables": [],
            "kind": "repository" if key.startswith("REPO:") else "external",
        }
        records.append(record)

    for record in records:
        override = overrides.get(record["asset_key"], {})
        for field in ("dataset", "link", "source", "publisher", "datatype", "size"):
            if override.get(field):
                record[field] = override[field]
        if override.get("themes"):
            record["themes"] = [value.strip() for value in override["themes"].split("|") if value.strip()]
        if override.get("variables"):
            record["variables"] = [value.strip() for value in override["variables"].split("|") if value.strip()]
        record["used_by"] = [
            {"code": code, "url": url} for code, url in record["used_by"]
        ]

    return sorted(records, key=lambda row: (row["kind"] != "mop", row["dataset"].lower()))


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "asset_id", "asset_key", "dataset", "link", "source", "publisher",
        "themes", "used_by", "datatype", "size", "variable_count", "variables",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{field: record.get(field, "") for field in fields},
                    "themes": " | ".join(record["themes"]),
                    "used_by": " | ".join(item["code"] for item in record["used_by"]),
                    "variable_count": len(record["variables"]) or "Not stated",
                    "variables": " | ".join(record["variables"]) or "Not stated",
                }
            )


BASE_STYLE = """<style>body{font-family:Arial,sans-serif;margin:2rem;color:#17212b}h1{margin-bottom:.35rem}.sub{margin:0 0 1rem;color:#52616b}.views,.filters,#pagination{display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;margin:0 0 1rem}button,select,input{padding:.5rem .65rem;font-size:.92rem}button{border:1px solid #073b4c;background:#fff;color:#073b4c;border-radius:.2rem;cursor:pointer}button.active{background:#073b4c;color:#fff}input{width:22rem;max-width:100%}table{border-collapse:collapse;width:100%;font-size:.88rem;table-layout:fixed}th,td{border:1px solid #d6dde3;padding:.55rem;vertical-align:top;text-align:left}th{background:#073b4c;color:#fff;position:sticky;top:0}th:nth-child(1),td:nth-child(1){width:8%}th:nth-child(2),td:nth-child(2){width:30%;overflow-wrap:anywhere}th:nth-child(3),td:nth-child(3){width:17%;overflow-wrap:anywhere}th:nth-child(4),td:nth-child(4){width:12%}th:nth-child(5),td:nth-child(5){width:13%}th:nth-child(6),td:nth-child(6){width:10%}body.mode-mop .source-col,body.mode-mop .source-filter,body.mode-selected .theme-col,body.mode-selected .theme-filter{display:none}tr:nth-child(even){background:#f7fafb}a{color:#075985}.tooltip{position:relative;cursor:help;border-bottom:1px dotted #075985}.tooltiptext{visibility:hidden;min-width:18rem;max-width:30rem;max-height:38rem;overflow-y:auto;background:#17212b;color:#fff;text-align:left;border-radius:.25rem;padding:.6rem;position:absolute;z-index:1;left:0;bottom:125%;font-weight:normal;line-height:1.45;white-space:nowrap}.tooltip:hover .tooltiptext{visibility:visible}@media(max-height:760px){.tooltiptext{max-height:calc(100vh - 5rem)}}</style>"""

DASHBOARD_STYLE = """<style>#data-asset-dashboard{margin-top:.5rem}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:0 0 1rem}.kpis article,.chart{border:1px solid #d6dde3;background:#fff;padding:1rem}.kpis span{display:block;color:#52616b;font-size:.85rem}.kpis strong{font-size:1.8rem;color:#073b4c}.insight-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.chart h2{font-size:1rem;margin:0 0 .8rem}.donut-wrap{display:flex;align-items:center;gap:1rem}.donut{width:9rem;height:9rem;border-radius:50%;display:grid;place-items:center;position:relative}.donut:after{content:'';width:5.6rem;height:5.6rem;background:#fff;border-radius:50%;position:absolute}.donut span{z-index:1;font-weight:bold;font-size:1.3rem}.bar-row{display:grid;grid-template-columns:minmax(8rem,1.4fr) minmax(7rem,2fr) 2rem;gap:.5rem;align-items:center;margin:.48rem 0}.bar-row span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track,.progress{background:#e2e8f0;height:.8rem}.bar-track i,.progress i{display:block;height:100%;background:#073b4c}.adoption-value{font-size:1.2rem;margin:1.3rem 0 .7rem}@media(max-width:760px){.kpis,.insight-grid{grid-template-columns:1fr}}</style>"""

POLISHED_STYLE = """<style>body{background:#f5f8fa}h1{font-size:2rem}.views{padding:.35rem;background:#e7eef1;border-radius:.65rem;width:max-content;max-width:100%}button{border-radius:.45rem;font-weight:600;transition:background .15s,color .15s}button.active{background:#0f766e;border-color:#0f766e}.filters select,.filters input{border:1px solid #cbd5e1;border-radius:.35rem;background:#fff}#data-asset-dashboard{margin-top:.5rem}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:0 0 1rem}.kpis article,.chart{border:1px solid #dbe4e9;background:#fff;padding:1.1rem;border-radius:.5rem;box-shadow:0 1px 2px rgba(15,23,42,.04)}.kpis article{border-top:4px solid #0f766e;display:grid;grid-template-columns:minmax(0,1fr) 4.8rem;align-items:center;min-height:4.3rem}.kpis article:nth-child(2){border-top-color:#2563eb}.kpis article:nth-child(3){border-top-color:#f59e0b}.kpis article:nth-child(4){border-top-color:#8b5cf6}.kpis span{display:block;color:#52616b;font-size:.88rem;line-height:1.3;text-align:left}.kpis strong{font-size:2rem;color:#073b4c;justify-self:center;text-align:center;transform:translateX(.25rem)}.insight-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.chart h2{font-size:1rem;margin:0 0 .8rem;color:#073b4c}.donut-wrap{display:flex;align-items:center;gap:1rem}.donut{width:9rem;height:9rem;border-radius:50%;display:grid;place-items:center;position:relative}.donut:after{content:'';width:5.6rem;height:5.6rem;background:#fff;border-radius:50%;position:absolute}.donut span{z-index:1;font-weight:bold;font-size:1.3rem;color:#073b4c}.bar-row{display:grid;grid-template-columns:minmax(8rem,1.4fr) minmax(7rem,2fr) 2rem;gap:.5rem;align-items:center;margin:.48rem 0}.bar-row span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track,.progress{background:#e2e8f0;height:.8rem;border-radius:.5rem;overflow:hidden}.bar-track i,.progress i{display:block;height:100%}.progress i{background:#0f766e}.bar-blue{background:#2563eb}.bar-orange{background:#f97316}.bar-purple{background:#8b5cf6}.adoption-value{font-size:1.2rem;margin:1.3rem 0 .7rem}.matrix-chart{margin-top:1rem}.matrix-wrap{overflow-x:auto}.matrix{width:100%;table-layout:auto;font-size:.82rem}.matrix th,.matrix td{text-align:center;min-width:5rem}.matrix th:first-child{text-align:left;min-width:11rem}.matrix td{color:#fff;font-weight:600}@media(max-width:760px){.kpis,.insight-grid{grid-template-columns:1fr}}</style>"""

REUSE_STYLE = """<style>.sub{display:none}.bar-row{grid-template-columns:minmax(6rem,.7fr) minmax(10rem,2.4fr) 1.8rem}.bar-track{max-width:24rem}.reuse-matrix-grid{display:grid;grid-template-columns:minmax(18rem,.8fr) minmax(28rem,1.7fr);gap:1rem;margin-top:1rem}.reuse-list ol{list-style:none;margin:0;padding:0}.reuse-list li{display:grid;grid-template-columns:2rem minmax(0,1fr) 2rem;gap:.55rem;align-items:center;padding:.54rem 0;border-bottom:1px solid #e2e8f0}.reuse-list li>span:nth-child(2){overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rank{display:grid;place-items:center;width:1.55rem;height:1.55rem;border-radius:50%;background:#ede9fe;color:#6d28d9;font-size:.78rem;font-weight:700}.reuse-list b{text-align:right;color:#6d28d9}.reuse-note{font-size:.8rem;color:#64748b;margin:.75rem 0 0}.matrix-chart{margin-top:0}.matrix{width:100%;table-layout:fixed;font-size:.74rem}.matrix th,.matrix td{min-width:0;padding:.7rem .35rem}.matrix th:first-child,.matrix td:first-child{width:26%;min-width:0}.matrix th:not(:first-child),.matrix td:not(:first-child){width:auto}@media(max-width:920px){.reuse-matrix-grid{grid-template-columns:1fr}}@media(max-width:760px){.kpis,.insight-grid{grid-template-columns:1fr}}</style>"""

VIEW_STYLE = """<style>.view-subtitles{min-height:2.6rem}.view-subtitle{display:none;margin:.55rem 0 1rem;color:#52616b;font-size:.95rem;line-height:1.45}.view-subtitle a{color:#075985}.mode-insights .subtitle-insights,.mode-mop .subtitle-mop,.mode-selected .subtitle-selected{display:block}.scope-note{margin:2rem 0 0;padding-top:1rem;border-top:1px solid #dbe4e9;color:#52616b;font-size:.85rem;line-height:1.45}</style>"""


def _load_use_case_domains(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["use_case_code"].strip().upper(): row["domain"].strip()
            for row in csv.DictReader(handle)
            if row.get("use_case_code", "").strip() and row.get("domain", "").strip()
        }


def _bars(title: str, values: dict[str, int], colour: str) -> str:
    maximum = max(values.values(), default=1)
    items = "".join(
        f'<div class="bar-row"><span title="{html.escape(label, quote=True)}">'
        f'{html.escape(label)}</span><div class="bar-track"><i class="{colour}" '
        f'style="width:{value / maximum * 100:.1f}%"></i></div><b>{value}</b></div>'
        for label, value in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:10]
    ) or "<p>Not stated</p>"
    return f'<section class="chart bar-chart"><h2>{html.escape(title)}</h2>{items}</section>'


def _render_insights(records: list[dict[str, Any]], domains: dict[str, str]) -> str:
    selected = [record for record in records if record["used_by"]]
    mop_selected = [record for record in selected if record["kind"] == "mop"]
    external_selected = [record for record in selected if record["kind"] != "mop"]
    all_mop = [record for record in records if record["kind"] == "mop"]
    adoption = len(mop_selected) / len(all_mop) * 100 if all_mop else 0
    mop_share = len(mop_selected) / len(selected) * 100 if selected else 0

    themes: dict[str, int] = defaultdict(int)
    use_cases: dict[str, int] = defaultdict(int)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in mop_selected:
        for theme in record["themes"]:
            themes[theme] += 1
        for use in record["used_by"]:
            domain = domains.get(use["code"], "Not stated")
            for theme in record["themes"]:
                matrix[domain][theme] += 1
    for record in selected:
        for use in record["used_by"]:
            use_cases[use["code"]] += 1

    reused = {
        record["dataset"]: len(record["used_by"])
        for record in selected
        if record["used_by"]
    }
    reused_items = "".join(
        f'<li><span class="rank">{index}</span><span title="{html.escape(name, quote=True)}">'
        f'{html.escape(name)}</span><b>{count}</b></li>'
        for index, (name, count) in enumerate(
            sorted(reused.items(), key=lambda item: (-item[1], item[0]))[:10],
            start=1,
        )
    ) or "<li>Not stated</li>"

    matrix_themes = sorted(
        {
            theme
            for record in all_mop
            for theme in record["themes"]
            if theme != "Not stated"
        }
    )
    matrix_max = max(
        (value for values in matrix.values() for value in values.values()),
        default=1,
    )
    matrix_header = "".join(f"<th>{html.escape(theme)}</th>" for theme in matrix_themes)
    matrix_rows = "".join(
        f"<tr><th>{html.escape(domain)}</th>"
        + "".join(
            f'<td style="background-color:rgba(13,110,253,'
            f'{matrix[domain].get(theme, 0) / matrix_max * .82:.2f})">'
            f'{matrix[domain].get(theme, 0) or ""}</td>'
            for theme in matrix_themes
        )
        + "</tr>"
        for domain in sorted(matrix)
    ) or "<tr><td>Not stated</td></tr>"

    return (
        '<section id="data-asset-dashboard"><div class="kpis">'
        f'<article><span>Total assets used</span><strong>{len(selected)}</strong></article>'
        f'<article><span>Used MOP assets</span><strong>{len(mop_selected)}</strong></article>'
        f'<article><span>External assets</span><strong>{len(external_selected)}</strong></article>'
        f'<article><span>MOP adoption rate</span><strong>{adoption:.1f}%</strong></article>'
        '</div><div class="insight-grid"><section class="chart">'
        '<h2>MOP vs external assets</h2><div class="donut-wrap">'
        f'<div class="donut" style="background:conic-gradient(#0f766e 0 {mop_share:.1f}%,'
        f'#f59e0b {mop_share:.1f}% 100%)"><span>{len(selected)}</span></div>'
        f'<div><p><b>{len(mop_selected)}</b> MOP assets</p>'
        f'<p><b>{len(external_selected)}</b> External assets</p></div></div></section>'
        '<section class="chart"><h2>MOP catalogue adoption</h2>'
        f'<p class="adoption-value"><b>{len(mop_selected)}</b> of {len(all_mop)} MOP datasets used</p>'
        f'<div class="progress"><i style="width:{adoption:.1f}%"></i></div>'
        f'<p>{adoption:.1f}% of the current catalogue</p></section>'
        + _bars("Assets by CoM Theme", themes, "bar-blue")
        + _bars("Use cases by number of datasets", use_cases, "bar-orange")
        + '</div><div class="reuse-matrix-grid"><section class="chart reuse-list">'
        f'<h2>Top reused datasets</h2><ol>{reused_items}</ol>'
        '<p class="reuse-note">Number indicates use cases using the dataset.</p></section>'
        '<section class="chart matrix-chart"><h2>Use Case Domain by City of Melbourne Theme</h2>'
        f'<div class="matrix-wrap"><table class="matrix"><thead><tr><th>Use case domain</th>'
        f'{matrix_header}</tr></thead><tbody>{matrix_rows}</tbody></table></div></section></div></section>'
    )


def _write_html(
    records: list[dict[str, Any]],
    path: Path,
    domains: dict[str, str],
) -> None:
    cell = lambda value: html.escape(str(value))
    source_options = "".join(
        f'<option value="{html.escape(value, quote=True)}">{cell(value)}</option>'
        for value in sorted({record["source"] for record in records})
    )
    theme_options = "".join(
        f'<option value="{html.escape(value.lower(), quote=True)}">{cell(value)}</option>'
        for value in sorted(
            {
                theme
                for record in records
                for theme in record["themes"]
                if theme != "Not applicable"
            }
        )
    )
    body: list[str] = []
    for record in records:
        dataset = (
            f'<a href="{html.escape(record["link"], quote=True)}" target="_blank" '
            f'rel="noopener">{cell(record["dataset"])}</a>'
        )
        used_by = "<br>".join(
            f'<a href="{html.escape(use["url"], quote=True)}" target="_blank" '
            f'rel="noopener">{cell(use["code"])}</a>'
            for use in record["used_by"]
        ) or "Not used in scanned use cases"
        variables = "<br>".join(cell(value) for value in record["variables"]) or "Not stated"
        variable_count = str(len(record["variables"])) if record["variables"] else "Not stated"
        type_size = (
            f'<b>Size:</b> {cell(record["size"])}<br><span class="tooltip">'
            f'<b>Variables:</b> {variable_count}<span class="tooltiptext">{variables}</span>'
            f'</span><br><b>Type:</b> {cell(record["datatype"])}'
        )
        attributes = (
            f'data-dataset="{html.escape(record["dataset"].lower(), quote=True)}" '
            f'data-use-cases="{html.escape(" ".join(use["code"] for use in record["used_by"]), quote=True)}" '
            f'data-source="{html.escape(record["source"], quote=True)}" '
            f'data-themes="{html.escape("|".join(theme.lower() for theme in record["themes"]), quote=True)}" '
            f'data-selected="{str(bool(record["used_by"])).lower()}" '
            f'data-mop="{str(record["kind"] == "mop").lower()}"'
        )
        body.append(
            f'<tr {attributes}><td>{cell(record["asset_id"])}</td><td>{dataset}</td>'
            f'<td class="source-col">{cell(record["source"])}</td>'
            f'<td class="theme-col">{"<br>".join(cell(theme) for theme in record["themes"])}</td>'
            f'<td>{used_by}</td><td class="type-col">{type_size}</td></tr>'
        )

    subtitles = (
        '<div class="view-subtitles">'
        '<p class="view-subtitle subtitle-insights">An overview of data assets used across Use Cases, '
        'including MOP Open Data adoption, reuse and theme coverage.</p>'
        '<p class="view-subtitle subtitle-mop">Explore the '
        '<a href="https://data.melbourne.vic.gov.au/explore/?sort=modified" target="_blank" '
        'rel="noopener">City of Melbourne Open Data catalogue</a>, including CoM Themes, '
        'data types and sizes.</p>'
        '<p class="view-subtitle subtitle-selected">Browse datasets used in FINALISED Use Cases, '
        'including MOP Open Data and external sources.</p></div>'
    )
    script = """<script>const pageSize=20;let page=1,mode='insights';function setMode(next){mode=next;page=1;document.body.className='mode-'+mode;document.querySelectorAll('[data-mode]').forEach(button=>button.classList.toggle('active',button.dataset.mode===mode));const insight=document.getElementById('data-asset-dashboard'),tableShell=document.getElementById('table-shell');const isInsights=mode==='insights';insight.hidden=!isInsights;tableShell.hidden=isInsights;if(isInsights)return;if(mode==='mop')document.getElementById('source').value='';else document.getElementById('theme').value='';filterRows()}function shownRows(){const q=document.getElementById('search').value.trim().toLowerCase(),source=document.getElementById('source').value,theme=document.getElementById('theme').value;return [...document.querySelectorAll('#catalogue tbody tr')].filter(row=>{const modeMatch=mode==='selected'?row.dataset.selected==='true':row.dataset.mop==='true';const searchMatch=!q||row.dataset.dataset.includes(q)||row.dataset.useCases.split(' ').some(code=>code.toLowerCase().startsWith(q));return modeMatch&&searchMatch&&(!source||row.dataset.source===source)&&(!theme||row.dataset.themes.split('|').includes(theme))})}function renderPage(){const all=[...document.querySelectorAll('#catalogue tbody tr')],shown=shownRows(),pages=Math.max(1,Math.ceil(shown.length/pageSize));page=Math.max(1,Math.min(page,pages));all.forEach(row=>row.style.display='none');shown.slice((page-1)*pageSize,page*pageSize).forEach(row=>row.style.display='');document.getElementById('page-info').textContent='Page '+page+' of '+pages+' ('+shown.length+' records)'}function filterRows(){page=1;renderPage()}function changePage(step){const pages=Math.max(1,Math.ceil(shownRows().length/pageSize));page=Math.max(1,Math.min(page+step,pages));renderPage()}renderPage();</script>"""
    content = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>MOP Data Asset Manage Platform</title>'
        + BASE_STYLE
        + DASHBOARD_STYLE
        + POLISHED_STYLE
        + REUSE_STYLE
        + VIEW_STYLE
        + '</head><body class="mode-insights"><h1>MOP Data Asset Manage Platform</h1>'
        '<p class="sub">A unified view of City of Melbourne Open Data, its use across MOP '
        'notebooks, and opportunities for data reuse.</p><div class="views">'
        '<button class="active" data-mode="insights" onclick="setMode(\'insights\')">'
        'Data Asset Dashboard</button><button data-mode="mop" onclick="setMode(\'mop\')">'
        'Open Data Catalogue</button><button data-mode="selected" onclick="setMode(\'selected\')">'
        'Usage</button></div>'
        + subtitles
        + _render_insights(records, domains)
        + '<div id="table-shell" hidden><div id="filters" class="filters">'
        '<input id="search" placeholder="Search Dataset name or Use Case code" oninput="filterRows()">'
        '<select id="source" class="source-filter" onchange="filterRows()">'
        f'<option value="">All Sources</option>{source_options}</select>'
        '<select id="theme" class="theme-filter" onchange="filterRows()">'
        f'<option value="">All CoM Themes</option>{theme_options}</select></div>'
        '<table id="catalogue"><thead><tr><th>Asset ID</th><th>Dataset</th>'
        '<th class="source-col">Source</th><th class="theme-col">CoM Theme</th>'
        '<th>Used by</th><th class="type-col">Size and Type</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table><div id="pagination">'
        '<button onclick="changePage(-1)">Previous</button><span id="page-info"></span>'
        '<button onclick="changePage(1)">Next</button></div></div>'
        '<footer class="scope-note">This platform is designed for FINALISED Use Cases and '
        'reflects the notebooks currently available in the repository.</footer>'
        + script
        + "</body></html>"
    )
    path.write_text(content, encoding="utf-8")


def build(config: BuildConfig | None = None) -> dict[str, Any]:
    """Build both distributable outputs and return a concise build summary."""
    if config is None:
        config = BuildConfig(project_dir=Path(__file__).resolve().parent)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = list_finalised_notebooks(config)
    findings = [
        finding
        for notebook in notebooks
        for finding in scan_notebook(notebook, config)
    ]
    city_catalogue = fetch_city_catalogue()
    records = build_records(findings, city_catalogue, _load_overrides(config.overrides_file))
    _assign_asset_ids(records, config.asset_ids_file)

    csv_path = config.output_dir / "Data_Asset_Catalogue.csv"
    html_path = config.output_dir / "Data_Asset_Catalogue.html"
    _write_csv(records, csv_path)
    _write_html(records, html_path, _load_use_case_domains(config.use_case_domains_file))

    used = [record for record in records if record["used_by"]]
    summary = {
        "source": f"github.com/{config.github_owner}/{config.github_repository}/{config.github_branch}/{config.finalised_path}",
        "notebooks_scanned": len(notebooks),
        "use_cases": len({item["code"] for row in used for item in row["used_by"]}),
        "catalogue_datasets": len(city_catalogue),
        "used_assets": len(used),
        "html": str(html_path),
        "csv": str(csv_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
