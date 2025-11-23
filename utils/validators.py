"""
Input validation utilities.
"""
import re
from typing import Optional


def validate_comment_id(comment_id: str) -> bool:
    """Validate comment ID format."""
    if not comment_id or len(comment_id) < 8:
        return False
    # Comment IDs are typically alphanumeric hashes
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', comment_id))


def validate_user_id(user_id: str) -> bool:
    """Validate user ID format."""
    if not user_id or len(user_id) < 8:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', user_id))


def sanitize_text(text: Optional[str], max_length: int = 1000) -> str:
    """Sanitize and truncate text input."""
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(' ', 1)[0] + '...'

    return sanitized.strip()


def validate_url(url: Optional[str]) -> bool:
    """Validate URL format."""
    if not url:
        return False

    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return bool(url_pattern.match(url))

