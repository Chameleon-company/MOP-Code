import os
import sys
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter

def read_notebook_file(notebook_path: Path):
    """Read a Jupyter notebook file and return the notebook object."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def convert_notebook_to_html(notebook) -> str:
    """Convert a notebook object to HTML string."""
    html_exporter = HTMLExporter()
    html_exporter.exclude_input_prompt = True
    html_exporter.exclude_output_prompt = True
    body, _ = html_exporter.from_notebook_node(notebook)
    return body

def write_html_file(html_content: str, output_path: Path):
    """Write HTML content to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def convert_notebooks_to_html(input_dir: str, output_dir: str):
    """Convert all .ipynb files in input_dir to HTML and save them to output_dir."""
    source = Path(input_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    for notebook_file in source.glob("*.ipynb"):
        notebook = read_notebook_file(notebook_file)
        html_content = convert_notebook_to_html(notebook)
        output_file = target / (notebook_file.stem + ".html")
        write_html_file(html_content, output_file)
        print(f"✔ Converted: {notebook_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_notebooks_to_html(input_folder, output_folder)