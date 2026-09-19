"""Escape untrusted text before it reaches a terminal or log line."""

import re

_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _escape_match(match: re.Match) -> str:
    return f"\\x{ord(match.group(0)):02x}"


def escape_control_characters(text: str) -> str:
    """Replace ASCII/C1 control characters (ESC, newlines, ...) with visible \\xNN escapes."""
    return _CONTROL_CHARACTER_PATTERN.sub(_escape_match, text)
