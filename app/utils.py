import hashlib

from app.db import DocumentType


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for the given content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_unique_identifier_hash(
    document_type: DocumentType,
    unique_identifier: str,
) -> str:
    """Generate SHA-256 hash for a unique document identifier."""
    combined_data = f"{document_type.value}:{unique_identifier}"
    return hashlib.sha256(combined_data.encode("utf-8")).hexdigest()