import re


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra spaces and line breaks.

    Parameters:
        text (str): Raw extracted text

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Replace multiple line breaks with a space
    text = re.sub(r"\n+", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()