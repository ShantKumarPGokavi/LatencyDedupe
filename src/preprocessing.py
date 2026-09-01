import re
import string

def clean_text(text: str) -> str:
    """Basic text normalization for question matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\?]", "", text)  # Keep words, spaces, and question marks
    text = re.sub(r"\s+", " ", text)
    return text