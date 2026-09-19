"""End-to-end CLI runs with a scripted model: stage wiring, persistence, exit codes."""

import json

import pytest

import research_agent
from conftest import FakeOllamaClient
from ollama_client import OllamaConnectionError
from pipeline import LessonStorageError
from user_profile import ProfileError


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Run main() in an isolated cwd/profile with scripted stdin and model outcomes."""
    profile_path = tmp_path / "config" / "user_profile.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BRAIN_SPARK_PROFILE", str(profile_path))
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)

    def run(stdin_lines, outcomes):
        answers = iter(stdin_lines)
        monkeypatch.setattr("builtins.input", lambda _label="": next(answers))
        fake_client = FakeOllamaClient(outcomes)
        monkeypatch.setattr(research_agent, "OllamaClient", lambda *args, **kwargs: fake_client)
        return research_agent.main(), fake_client

    run.profile_path = profile_path
    run.directory = tmp_path
    return run


def test_cli_first_and_returning_runs(cli, capsys):
    exit_code, client = cli(["Alex", "How do black holes form?"], ["FACTS", "# Lesson\n\nBody", "stellar-collapse"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Nice to meet you, Alex!" in output
    assert "Written to: stellar-collapse.md" in output and "Done! :)" in output
    assert (cli.directory / "stellar-collapse.md").read_text() == "# Lesson\n\nBody\n"
    assert json.loads(cli.profile_path.read_text()) == {"name": "Alex"}
    assert "How do black holes form?" in client.calls[0][0]
    assert "FACTS" in client.calls[1][0]

    exit_code, _ = cli(["Another topic"], ["F2", "L2", "second"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Welcome back, Alex!" in output and "What's your name" not in output
    assert (cli.directory / "second.md").read_text() == "L2\n"


def test_cli_reprompts_on_invalid_input(cli, capsys):
    exit_code, _ = cli(["", "bad;name", "Alex", "   ", "topic"], ["F", "L", "name"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("Try again.") == 3


def test_cli_config_error_exit_code(cli, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "zero")
    exit_code, _ = cli([], [])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OLLAMA_TIMEOUT_SECONDS" in captured.err and "Done!" not in captured.out


def test_cli_transport_error_exit_code(cli, capsys):
    exit_code, _ = cli(["Alex", "topic"], [OllamaConnectionError("Cannot reach Ollama")])
    captured = capsys.readouterr()
    assert exit_code == 3
    assert "Cannot reach Ollama" in captured.err and "Done!" not in captured.out
    assert list(path for path in cli.directory.iterdir() if path.suffix == ".md") == []


def test_cli_storage_error_exit_code(cli, monkeypatch, capsys):
    monkeypatch.setattr(research_agent, "save_name", lambda *_: (_ for _ in ()).throw(ProfileError("disk full")))
    exit_code, _ = cli(["Alex"], [])
    captured = capsys.readouterr()
    assert exit_code == 4
    assert "disk full" in captured.err and "Done!" not in captured.out


def test_cli_write_failure_exit_code(cli, monkeypatch, capsys):
    monkeypatch.setattr(research_agent, "write_lesson", lambda *_: (_ for _ in ()).throw(
        LessonStorageError("Could not write")))
    exit_code, _ = cli(["Alex", "topic"], ["F", "L", "name"])
    captured = capsys.readouterr()
    assert exit_code == 4 and "Done!" not in captured.out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_cli_interruption_exit_code(cli, monkeypatch, capsys, interruption):
    monkeypatch.setattr("builtins.input", lambda _label="": (_ for _ in ()).throw(interruption()))
    exit_code = research_agent.main()
    captured = capsys.readouterr()
    assert exit_code == 130
    assert "Bye!" in captured.out and "Done!" not in captured.out


def test_filename_failure_preserves_completed_lesson(cli, capsys):
    exit_code, _ = cli(["Alex", "How do Black Holes form?"], ["F", "# Saved", OllamaConnectionError("down")])
    assert exit_code == 0
    written = cli.directory / "how-do-black-holes-form.md"
    assert written.read_text() == "# Saved\n"
    assert "Written to: how-do-black-holes-form.md" in capsys.readouterr().out


def test_cli_truncated_twice_writes_nothing(cli, capsys):
    from conftest import TRUNCATED
    exit_code, _ = cli(["Alex", "topic"], [TRUNCATED, TRUNCATED])
    captured = capsys.readouterr()
    assert exit_code == 5
    assert "incomplete after retry" in captured.err
    assert [path for path in cli.directory.iterdir() if path.suffix == ".md"] == []
