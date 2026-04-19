\# Knowledge Extraction from Urban Planning Documents



This project extracts key planning knowledge from urban planning documents and converts unstructured text into structured outputs such as JSON and CSV.



\## Project Goal

The goal of this project is to identify and extract important planning concepts from government planning documents, including:

\- term

\- definition

\- purpose\_or\_function

\- permit\_related\_information

\- authority\_involved

\- related\_components

\- source\_section



\## Folder Structure

\- `data/raw/` stores original PDF documents

\- `data/processed/` stores extracted and cleaned text

\- `outputs/` stores JSON and CSV results

\- `src/` stores Python source code



\## How to Run

1\. Put your PDF files into `data/raw/`

2\. Run the main script:

&#x20;  ```bash

&#x20;  python src/main.py

