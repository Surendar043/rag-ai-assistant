"""Helper functions shared across the project."""


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    return " ".join(text.split())
