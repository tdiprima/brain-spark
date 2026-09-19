"""Output boundaries: terminal escaping and one-line JSON logs."""

import json
import logging

import pytest

import research_agent
from config import JsonLogFormatter, configure_logging
from sanitize import escape_control_characters

HOSTILE = "\x1b[2Jwipe\rline\nfeed\x7fdel\x9bcsi\x00nul"


def test_sanitizer_escapes_every_control_character():
    escaped = escape_control_characters(HOSTILE)
    assert escaped == "\\x1b[2Jwipe\\x0dline\\x0afeed\\x7fdel\\x9bcsi\\x00nul"
    assert not any(ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in escaped)


@pytest.mark.parametrize("text", ["", "plain ascii", "café ☕ 黑洞 emoji 🧠✨", "tabs are\ttext? no"])
def test_sanitizer_preserves_unicode_text(text):
    expected = text.replace("\t", "\\x09")
    assert escape_control_characters(text) == expected


def test_backend_error_is_safe_on_stderr(capsys):
    research_agent.show_error(HOSTILE)
    captured = capsys.readouterr().err
    without_app_colors = captured.replace(research_agent.RED, "").replace(research_agent.RESET, "")
    assert "\x1b" not in without_app_colors
    assert "\r" not in without_app_colors and "\x7f" not in without_app_colors and "\x9b" not in without_app_colors
    assert without_app_colors.count("\n") == 1 and without_app_colors.endswith("\n")
    assert "\\x1b[2J" in captured and "\\x0a" in captured
    assert captured.startswith(research_agent.RED + "Error: ")


def test_log_record_is_one_json_line():
    formatter = JsonLogFormatter()
    message = 'quote " and \n newline \x1b esc'
    try:
        raise RuntimeError("boom\nsecond line")
    except RuntimeError:
        import sys
        record = logging.LogRecord("comp", logging.ERROR, "", 0, message, None, sys.exc_info())
    line = formatter.format(record)
    assert "\n" not in line and "\x1b" not in line
    parsed = json.loads(line)
    assert parsed["event"] == message
    assert parsed["level"] == "ERROR" and parsed["component"] == "comp"
    assert "RuntimeError: boom" in parsed["exception"]
    assert set(parsed) == {"time", "level", "component", "event", "exception"}


def test_configure_logging_installs_json_handler():
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(handler.formatter, JsonLogFormatter) for handler in root.handlers)
