import re

def extract_clean_content(raw_text: str, target_format: str = "bibtex") -> str:
    """
    Strips conversational chat wrapper and extracts only clean,
    valid BibTeX citation blocks (@article, @book, etc.).
    """
    if not raw_text:
        return ""

    # 1. Look for markdown code blocks (e.g., ```bibtex ... ```)
    code_blocks = re.findall(r"```(?:bibtex|bib)?\s*([\s\S]*?)```", raw_text, re.IGNORECASE)
    if code_blocks:
        extracted = "\n\n".join(cb.strip() for cb in code_blocks if "@" in cb)
        if extracted:
            return extracted

    # 2. Fallback: Extract every discrete @type{...} block directly from raw text
    bibtex_entries = re.findall(r"(@[a-zA-Z]+\s*\{[\s\S]*?\n\})", raw_text)
    if bibtex_entries:
        return "\n\n".join(entry.strip() for entry in bibtex_entries)

    # 3. Last fallback: capture from the first '@' tag to the end
    raw_bib = re.search(r"(@[a-zA-Z]+\s*\{[\s\S]*)", raw_text)
    if raw_bib:
        # Strip potential trailing conversational text if enclosed in markdown
        cleaned = re.sub(r"```.*$", "", raw_bib.group(1), flags=re.DOTALL)
        return cleaned.strip()

    return raw_text.strip()