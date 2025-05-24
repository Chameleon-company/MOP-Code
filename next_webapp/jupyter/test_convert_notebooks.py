import unittest
import tempfile
from pathlib import Path
import nbformat

from convert_notebooks import (
    read_notebook_file,
    convert_notebook_to_html,
    write_html_file,
    convert_notebooks_to_html
)

class TestNotebookConversion(unittest.TestCase):

    def setUp(self):
        # Create a minimal notebook
        self.notebook_content = nbformat.v4.new_notebook()
        self.notebook_content.cells.append(nbformat.v4.new_markdown_cell("## Test Notebook"))

        # Create temporary directories
        self.temp_input_dir = tempfile.TemporaryDirectory()
        self.temp_output_dir = tempfile.TemporaryDirectory()
        self.input_path = Path(self.temp_input_dir.name) / "test_notebook.ipynb"

        # Write test notebook to file
        with open(self.input_path, 'w', encoding='utf-8') as f:
            nbformat.write(self.notebook_content, f)

    def tearDown(self):
        # Clean up temp directories
        self.temp_input_dir.cleanup()
        self.temp_output_dir.cleanup()

    def test_convert_notebooks_to_html(self):
        # Run the converter
        convert_notebooks_to_html(
            str(self.temp_input_dir.name),
            str(self.temp_output_dir.name)
        )

        output_file = Path(self.temp_output_dir.name) / "test_notebook.html"

        # Assert output file exists and contains expected content
        self.assertTrue(output_file.exists(), "HTML output file was not created.")
        content = output_file.read_text(encoding='utf-8')
        self.assertIn("Test Notebook", content, "HTML content does not contain expected notebook text.")

if __name__ == '__main__':
    unittest.main()
