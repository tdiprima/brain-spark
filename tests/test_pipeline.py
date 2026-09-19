"""Pipeline: topic validation, slugging, truncation recovery, collision-safe writes."""

import os
import threading
from pathlib import Path

import pytest

import pipeline
from config import MAX_TOPIC_LENGTH, RESEARCH_MAX_TOKENS, TEACH_MAX_TOKENS
from conftest import TRUNCATED
from ollama_client import OllamaConnectionError


@pytest.mark.parametrize("raw_topic", ["", "   ", "\t\n"])
def test_validate_topic_rejects_empty(raw_topic):
    with pytest.raises(ValueError, match="empty"):
        pipeline.validate_topic(raw_topic)


def test_validate_topic_length_boundary():
    assert len(pipeline.validate_topic("x" * MAX_TOPIC_LENGTH)) == MAX_TOPIC_LENGTH
    with pytest.raises(ValueError, match="at most"):
        pipeline.validate_topic("x" * (MAX_TOPIC_LENGTH + 1))


def test_validate_topic_collapses_whitespace():
    assert pipeline.validate_topic("  black \n holes  ") == "black holes"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stellar-collapse", "stellar-collapse"),
        ("../../etc/passwd", "etc-passwd"),
        ("/absolute/path", "absolute-path"),
        ("C:\\windows\\system32", "c-windows-system32"),
        ("first line\nsecond line", "first-line"),
        ("  Black Holes!  ", "black-holes"),
        ("", "lesson"),
        ("   \n\n", "lesson"),
        ("---", "lesson"),
        ("\x1b[31mred\x1b[0m", "31mred-0m"),
        ("黑洞", "lesson"),
        ("café au lait", "caf-au-lait"),
    ],
)
def test_slugify_is_safe_and_path_free(raw, expected):
    slug = pipeline.slugify(raw)
    assert slug == expected
    assert "/" not in slug and "\\" not in slug and ".." not in slug


def test_slugify_bounds_length():
    slug = pipeline.slugify("word-" * 100)
    assert len(slug) <= pipeline.MAX_FILENAME_LENGTH
    assert not slug.endswith("-")


def test_slugged_filename_stays_inside_output_directory(tmp_path):
    for hostile in ["../../escape", "/tmp/escape", "..", "~/escape"]:
        written = pipeline.write_lesson(tmp_path, pipeline.slugify(hostile), "body")
        assert written.parent == tmp_path
        assert written.name.endswith(".md")


def test_truncation_recovery_is_bounded(fake_client_factory, tmp_path):
    client = fake_client_factory([TRUNCATED, "complete facts"])
    assert pipeline.research(client, "topic") == "complete facts"
    assert [budget for _, budget in client.calls] == [RESEARCH_MAX_TOKENS, RESEARCH_MAX_TOKENS * 2]

    client = fake_client_factory([TRUNCATED, TRUNCATED, "never used"])
    with pytest.raises(pipeline.PipelineError, match="incomplete after retry"):
        pipeline.teach(client, "topic", "facts")
    assert [budget for _, budget in client.calls] == [TEACH_MAX_TOKENS, TEACH_MAX_TOKENS * 2]
    assert list(tmp_path.iterdir()) == []


def test_research_and_teach_pass_topic_and_facts_into_prompts(fake_client_factory):
    client = fake_client_factory(["facts", "lesson"])
    pipeline.research(client, "quantum foam")
    pipeline.teach(client, "quantum foam", "FACT-LIST")
    assert "quantum foam" in client.calls[0][0]
    assert "FACT-LIST" in client.calls[1][0] and "quantum foam" in client.calls[1][0]


def test_suggest_filename_falls_back_to_topic_slug(fake_client_factory):
    client = fake_client_factory([OllamaConnectionError("down")])
    assert pipeline.suggest_filename(client, "How do Black Holes form?") == "how-do-black-holes-form"
    client = fake_client_factory(["../Stellar Collapse\nextra"])
    assert pipeline.suggest_filename(client, "ignored") == "stellar-collapse"


def test_write_lesson_never_overwrites_existing(tmp_path):
    existing = tmp_path / "x.md"
    existing.write_text("original")
    written = pipeline.write_lesson(tmp_path, "x", "new")
    assert written == tmp_path / "x-2.md"
    assert existing.read_text() == "original"
    assert written.read_text() == "new\n"


def test_concurrent_lessons_preserve_both_contents(tmp_path):
    existing = tmp_path / "shared.md"
    existing.write_text("pre-existing")
    writer_count = 8
    barrier = threading.Barrier(writer_count)
    results = {}

    def write(index):
        barrier.wait()
        results[index] = pipeline.write_lesson(tmp_path, "shared", f"lesson-{index}")

    threads = [threading.Thread(target=write, args=(index,)) for index in range(writer_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert existing.read_text() == "pre-existing"
    assert len(set(results.values())) == writer_count
    for index, path in results.items():
        assert path.read_text() == f"lesson-{index}\n"


def test_write_lesson_collision_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_FILENAME_ATTEMPTS", 3)
    for name in ["x.md", "x-2.md", "x-3.md"]:
        (tmp_path / name).write_text("taken")
    with pytest.raises(pipeline.PipelineError, match="Too many existing files"):
        pipeline.write_lesson(tmp_path, "x", "body")
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["x-2.md", "x-3.md", "x.md"]


def test_write_lesson_surfaces_io_failure(tmp_path):
    with pytest.raises(pipeline.PipelineError, match="Could not write"):
        pipeline.write_lesson(tmp_path / "missing-directory", "x", "body")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_write_lesson_permission_denied(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(pipeline.PipelineError, match="Could not write"):
            pipeline.write_lesson(locked, "x", "body")
    finally:
        locked.chmod(0o700)


def test_write_lesson_handles_large_content(tmp_path):
    big_lesson = "word " * 200_000
    written = pipeline.write_lesson(tmp_path, "big", big_lesson)
    assert written.stat().st_size == len(big_lesson) + 1
