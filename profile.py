"""User profile storage: remembers the user's name between sessions."""

import json
import logging
import re
from pathlib import Path

from config import MAX_NAME_LENGTH

logger = logging.getLogger("profile")

_NAME_PATTERN = re.compile(r"^[\w .'\-]+$", re.UNICODE)


class ProfileError(Exception):
    """Raised when the profile file cannot be read or written."""


def validate_name(raw_name: str) -> str:
    """Return a cleaned name or raise ValueError with a clear reason."""
    name = raw_name.strip()
    if not name:
        raise ValueError("Name must not be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Name must be at most {MAX_NAME_LENGTH} characters")
    if not _NAME_PATTERN.match(name):
        raise ValueError("Name may only contain letters, digits, spaces, dots, hyphens and apostrophes")
    return name


def load_name(profile_path: str) -> str | None:
    """Return the saved name, or None when no valid profile exists."""
    path = Path(profile_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("profile_unreadable path=%s error=%s", path, error)
        return None
    if not isinstance(data, dict):
        logger.warning("profile_malformed path=%s", path)
        return None
    raw_name = data.get("name")
    if not isinstance(raw_name, str):
        logger.warning("profile_invalid_name path=%s", path)
        return None
    try:
        return validate_name(raw_name)
    except ValueError:
        logger.warning("profile_invalid_name path=%s", path)
        return None


def save_name(profile_path: str, name: str) -> None:
    """Persist the validated name to the profile file."""
    clean_name = validate_name(name)
    path = Path(profile_path)
    try:
        path.write_text(json.dumps({"name": clean_name}, indent=2), encoding="utf-8")
    except OSError as error:
        logger.error("profile_write_failed path=%s error=%s", path, error)
        raise ProfileError(f"Could not save profile to {path}: {error}") from error
