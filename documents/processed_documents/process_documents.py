import os
import fitz  # PyMuPDF

# 📂 Folder paths
RAW_DIR = "../raw_documents"
PROCESSED_DIR = "."

# Create output folder if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


def convert_pdf_to_md(pdf_path, output_path):
    try:
        # Skip empty files
        if os.path.getsize(pdf_path) == 0:
            print(f"⚠️ Skipping empty file: {pdf_path}")
            return

        doc = fitz.open(pdf_path)
        text = ""

        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            if not page_text.strip():
                continue

            text += f"\n\n## Page {page_num + 1}\n"
            text += page_text

        if not text.strip():
            print(f"⚠️ No readable text in: {pdf_path}")
            return

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")


def process_all_pdfs():
    print("🚀 Starting conversion...\n")

    for file in os.listdir(RAW_DIR):
        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(RAW_DIR, file)
            md_name = file.replace(".pdf", ".md")
            output_path = os.path.join(PROCESSED_DIR, md_name)

            print(f"📄 Processing: {file}")

            convert_pdf_to_md(pdf_path, output_path)

            print(f"✅ Created: {md_name}\n")

    print("🎉 All PDFs converted successfully!")


if __name__ == "__main__":
    process_all_pdfs()