import os
import logging
from src.extract_text import extract_all_pdfs
from src.preprocess_text import clean_text
from src.knowledge_extraction import extract_knowledge
from src.save_output import save_as_json, save_as_csv, ensure_folder

# --- PROFESSIONAL LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/processing.log"), # Saves a log file for the tutor to see
        logging.StreamHandler() # Prints to your terminal
    ]
)

def generate_quality_report(results):
    """Generates a summary of extraction success for the meeting."""
    total = len(results)
    if total == 0:
        return

    # Count how many times we successfully found data instead of "Not Found"
    zoning_count = sum(1 for r in results if r.get('Zoning') != "Not Found")
    height_count = sum(1 for r in results if r.get('Max_Building_Height') != "Not Found")
    parking_count = sum(1 for r in results if r.get('Proposed_Parking') != "Not Found")

    print("\n" + "="*40)
    print("      DATA QUALITY REPORT")
    print("="*40)
    print(f"Total Documents Processed: {total}")
    print(f"Zoning Identified:        {zoning_count} / {total}")
    print(f"Height Identified:        {height_count} / {total}")
    print(f"Parking Identified:       {parking_count} / {total}")
    print("="*40 + "\n")

def main():
    raw_data_folder = "data/raw"
    processed_data_folder = "data/processed"
    output_folder = "outputs"

    ensure_folder(processed_data_folder)
    ensure_folder(output_folder)

    logging.info("Starting PDF extraction process...")
    extracted_documents = extract_all_pdfs(raw_data_folder)

    if not extracted_documents:
        logging.warning("No PDF files found in data/raw!")
        return

    all_results = []

    for filename, raw_text in extracted_documents.items():
        logging.info(f"Processing: {filename}")
        cleaned_text = clean_text(raw_text)

        # Save cleaned text for transparency
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(processed_data_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Extract structured data
        results = extract_knowledge(cleaned_text, filename)
        all_results.extend(results)

    # Save structured outputs
    save_as_json(all_results, os.path.join(output_folder, "extracted_results.json"))
    save_as_csv(all_results, os.path.join(output_folder, "extracted_results.csv"))

    logging.info("Processing complete. Files saved to outputs/")
    
    # Show the tutor the quality of your coding
    generate_quality_report(all_results)


if __name__ == "__main__":
    main()