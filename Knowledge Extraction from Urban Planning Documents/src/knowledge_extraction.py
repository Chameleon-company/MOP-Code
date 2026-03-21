import re


TARGET_TERMS = [
    "planning scheme",
    "zone",
    "overlay",
    "planning permit",
    "responsible authority",
    "planning authority",
    "schedule"
]


def extract_sentence_for_term(text: str, term: str) -> str:
    """
    Find a sentence containing the target term.

    Parameters:
        text (str): Cleaned document text
        term (str): Target planning term

    Returns:
        str: A matching sentence or 'Not specified'
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        if term.lower() in sentence.lower():
            return sentence.strip()

    return "Not specified"


def extract_knowledge(text: str, document_name: str) -> list:
    """
    Extract structured planning knowledge from the text.

    Parameters:
        text (str): Cleaned document text
        document_name (str): Name of the source document

    Returns:
        list: List of extracted records
    """
    results = []

    for term in TARGET_TERMS:
        sentence = extract_sentence_for_term(text, term)

        record = {
            "document_name": document_name,
            "term": term,
            "definition": sentence,
            "purpose_or_function": "To be refined",
            "permit_related_information": "To be refined",
            "authority_involved": "To be refined",
            "related_components": "To be refined",
            "source_section": "Not specified"
        }

        results.append(record)

    return results