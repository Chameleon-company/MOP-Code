import os
from src.extract_text import extract_all_pdfs
from src.preprocess_text import clean_text
from src.knowledge_extraction import extract_knowledge
from src.save_output import save_as_json, save_as_csv, ensure_folder


def main():
    raw_data_folder = "data/raw"
    processed_data_folder = "data/processed"
    output_folder = "outputs"

    ensure_folder(processed_data_folder)
    ensure_folder(output_folder)

    extracted_documents = extract_all_pdfs(raw_data_folder)

    all_results = []

    for filename, raw_text in extracted_documents.items():
        cleaned_text = clean_text(raw_text)

        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(processed_data_folder, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        results = extract_knowledge(cleaned_text, filename)
        all_results.extend(results)

    save_as_json(all_results, os.path.join(output_folder, "extracted_results.json"))
    save_as_csv(all_results, os.path.join(output_folder, "extracted_results.csv"))

    print("Processing complete.")
    print(f"Processed {len(extracted_documents)} PDF file(s).")
    print("Results saved in the outputs folder.")


if __name__ == "__main__":
    main()