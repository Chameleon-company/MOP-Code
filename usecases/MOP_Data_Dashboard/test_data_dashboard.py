"""Offline regressions for missing FINALISED cases and truthful totals."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_dashboard as dashboard


class DashboardTests(unittest.TestCase):
    def scan(self, cells):
        notebook = {
            "path": "usecases/FINALISED/UC00999_example.ipynb",
            "download_url": "https://example.invalid/notebook",
            "html_url": "https://example.invalid/UC00999",
        }
        with patch.object(dashboard, "_request", return_value=json.dumps({"cells": cells})):
            return dashboard.scan_notebook(notebook, dashboard.BuildConfig(Path(".")))

    def test_heading_and_links_share_a_cell(self):
        for heading in (
            '<div class="usecase-unnumbered-heading"><b>Data Sets</b></div>',
            '## Data Sets',
            '<h2>Data Sets</h2>',
        ):
            with self.subTest(heading=heading):
                findings = self.scan([{"cell_type": "markdown", "source": heading + '\n'
                    '[Dwellings](https://data.melbourne.vic.gov.au/explore/dataset/residential-dwellings/)\n'
                    '## Further reading\n'
                    '[Unrelated](https://example.invalid/unrelated.csv)'}])
                self.assertEqual({row["asset_key"] for row in findings}, {"MOP:residential-dwellings"})

    def test_heading_and_links_in_separate_cells(self):
        findings = self.scan([
            {"cell_type": "markdown", "source": "Data Sets"},
            {"cell_type": "markdown", "source": '<a href="https://example.invalid/used.csv">Used</a>'},
            {"cell_type": "markdown", "source": '<div class="usecase-section-heading">Analysis</div>'},
            {"cell_type": "markdown", "source": '[Unused](https://example.invalid/unused.csv)'},
        ])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["link"], "https://example.invalid/used.csv")

    def test_air_quality_variable_arguments(self):
        findings = self.scan([{"cell_type": "code", "source":
            "dataset_id_1 = 'argyle-square-air-quality'\n"
            "dataset_id_2 = 'development-activity-monitor'\n"
            "AirQuality_df = API_Unlimited(dataset_id_1)\n"
            "ActivityMonitor_df = API_Unlimited(dataset_id_2)"}])
        self.assertEqual({row["asset_key"] for row in findings}, {
            "MOP:argyle-square-air-quality", "MOP:development-activity-monitor",
        })

    def test_cross_cell_variables_and_reassignment(self):
        findings = self.scan([
            {"cell_type": "code", "source": "dataset_id = 'residential-dwellings'"},
            {"cell_type": "code", "source": "df = API_Unlimited(dataset_id)"},
            {"cell_type": "code", "source": "dataset_id = 'soil-sensor-locations'\ndataset_id = input()"},
            {"cell_type": "code", "source": "df = API_Unlimited(dataset_id)"},
        ])
        self.assertEqual({row["asset_key"] for row in findings}, {"MOP:residential-dwellings"})

    def test_existing_helpers_and_keyword_arguments(self):
        self.assertEqual(dashboard._static_dataset_ids(
            "dataset_id = 'residential-dwellings'\n"
            "a = fetch_data(base_url, dataset_id)\n"
            "b = fetch_melbourne_dataset('soil-sensor-locations')\n"
            "c = collect_data(dataset_id='argyle-square-air-quality')\n"
            "d = fetch_geojson_dataset_API('blocks-for-census-of-land-use-and-employment-clue')\n"
        ), {
            "residential-dwellings", "soil-sensor-locations", "argyle-square-air-quality",
            "blocks-for-census-of-land-use-and-employment-clue",
        })

    def test_does_not_use_unrelated_or_function_local_strings(self):
        self.assertEqual(dashboard._static_dataset_ids(
            "unrelated = 'not-a-dataset'\n"
            "def unused():\n    dataset_id = 'not-global'\n"
            "df = API_Unlimited(dataset_id)\n"
        ), set())
        self.assertEqual(dashboard._static_dataset_ids("%matplotlib inline"), set())

    def test_unmatched_notebook_still_counts_and_is_reported(self):
        notebooks = [
            {"path": f"usecases/FINALISED/{code}_example.ipynb"}
            for code in ("UC00998", "UC00999")
        ]
        finding = {
            "asset_key": "MOP:example-data", "dataset": "Example", "link": "https://example.invalid/data",
            "source": "Example", "publisher": "Example", "notebook_code": "UC00998",
            "notebook_url": "https://example.invalid/UC00998",
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = dashboard.BuildConfig(Path(temporary))
            with (
                patch.object(dashboard, "list_finalised_notebooks", return_value=notebooks),
                patch.object(dashboard, "scan_notebook", side_effect=[[finding], []]),
                patch.object(dashboard, "fetch_city_catalogue", return_value={}),
                patch.object(dashboard, "resolve_use_case_domains", return_value={}),
                patch.object(dashboard, "_logo_data_uri", return_value="data:image/png;base64,"),
                patch("builtins.print"),
            ):
                summary = dashboard.build(config)
            self.assertEqual(summary["use_cases"], 2)
            self.assertEqual(summary["use_cases_with_datasets"], 1)
            self.assertEqual(summary["use_cases_without_datasets"], ["UC00999"])
            self.assertEqual(json.loads((config.output_dir / "build-summary.json").read_text()), summary)
            page = (config.output_dir / "MOP Data Dashboard.html").read_text()
            self.assertIn('FINALISED Use Cases</span><strong>2</strong>', page)
            self.assertIn('No dataset references identified for: UC00999', page)

    def test_actual_finalised_notebooks(self):
        """Protect the actual UC00023/UC00053 formats without network or execution."""
        root = Path(__file__).resolve().parent.parent / "FINALISED"
        expectations = {
            "UC00023": {
                "MOP:residential-dwellings", "MOP:blocks-for-census-of-land-use-and-employment-clue",
                "MOP:cafes-and-restaurants-with-seating-capacity", "MOP:employment-by-block-by-clue-industry",
            },
            "UC00053": {"MOP:argyle-square-air-quality", "MOP:development-activity-monitor"},
        }
        for code, expected in expectations.items():
            with self.subTest(code=code):
                path = next(root.glob(f"{code}_*.ipynb"))
                findings = self.scan(json.loads(path.read_text(encoding="utf-8"))["cells"])
                self.assertEqual({row["asset_key"] for row in findings}, expected)


if __name__ == "__main__":
    unittest.main()
