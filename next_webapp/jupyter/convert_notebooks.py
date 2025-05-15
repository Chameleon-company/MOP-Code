import os
import sys
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter

def convert_notebooks_to_html(input_dir: str, output_dir: str):
    source = Path(input_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    for notebook_file in source.glob("*.ipynb"):
        with open(notebook_file, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        body, _ = html_exporter.from_notebook_node(notebook)

        output_file = target / (notebook_file.stem + ".html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(body)

        print(f"✔ Converted: {notebook_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_notebooks_to_html(input_folder, output_folder)
