"""Research -> teach -> filename -> write pipeline."""

import logging
import re
from pathlib import Path

from config import (
    FILENAME_MAX_TOKENS,
    MAX_TOPIC_LENGTH,
    RESEARCH_MAX_TOKENS,
    RESEARCH_WORD_LIMIT,
    TEACH_MAX_TOKENS,
    TEACH_WORD_LIMIT,
)
from ollama_client import OllamaClient

logger = logging.getLogger("pipeline")

MAX_FILENAME_LENGTH = 60
FALLBACK_FILENAME = "lesson"
_SLUG_CLEAN_PATTERN = re.compile(r"[^a-z0-9]+")


class PipelineError(Exception):
    """Raised when a lesson cannot be produced or written."""


def validate_topic(raw_topic: str) -> str:
    """Return a cleaned topic or raise ValueError with a clear reason."""
    topic = " ".join(raw_topic.split())
    if not topic:
        raise ValueError("Topic must not be empty")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Topic must be at most {MAX_TOPIC_LENGTH} characters")
    return topic


def build_research_prompt(topic: str) -> str:
    return (
        f"Give concise, factual information about: {topic}\n\n"
        f"Rules:\n"
        f"- Stay under {RESEARCH_WORD_LIMIT} words.\n"
        f"- Use plain bullet points of key facts.\n"
        f"- No introduction, no conclusion, no opinions."
    )


def build_teach_prompt(topic: str, research: str) -> str:
    return (
        f"You are an enthusiastic 10th-grade teacher. Using ONLY the facts below, "
        f"write a markdown lesson about: {topic}\n\n"
        f"Facts:\n{research}\n\n"
        f"Rules:\n"
        f"- Stay under {TEACH_WORD_LIMIT} words.\n"
        f"- Start with a '# ' title.\n"
        f"- Use real-life analogies and comparisons to similar concepts.\n"
        f"- End with one short mind-blowing fact.\n"
        f"- Warm, curious tone. Finish every sentence."
    )


def build_filename_prompt(topic: str) -> str:
    return (
        f"Suggest a short filename (2-4 lowercase words joined by hyphens, no extension) "
        f"for a lesson about: {topic}\n"
        f"Reply with the filename only."
    )


def slugify(raw_text: str) -> str:
    """Turn model output into a safe, path-free filename stem."""
    first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    slug = _SLUG_CLEAN_PATTERN.sub("-", first_line.lower()).strip("-")
    slug = slug[:MAX_FILENAME_LENGTH].strip("-")
    return slug or FALLBACK_FILENAME


def unique_markdown_path(directory: Path, stem: str) -> Path:
    """Return <stem>.md or <stem>-N.md so existing files are never overwritten."""
    candidate = directory / f"{stem}.md"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}.md"
        counter += 1
    return candidate


def research(client: OllamaClient, topic: str) -> str:
    return client.generate(build_research_prompt(topic), RESEARCH_MAX_TOKENS)


def teach(client: OllamaClient, topic: str, facts: str) -> str:
    return client.generate(build_teach_prompt(topic, facts), TEACH_MAX_TOKENS)


def suggest_filename(client: OllamaClient, topic: str) -> str:
    """Ask the model for a name; fall back to a slug of the topic on failure."""
    try:
        return slugify(client.generate(build_filename_prompt(topic), FILENAME_MAX_TOKENS))
    except Exception as error:  # noqa: BLE001 - any model failure degrades to topic slug
        logger.warning("filename_generation_failed error=%s", error)
        return slugify(topic)


def write_lesson(directory: Path, stem: str, lesson: str) -> Path:
    """Write the lesson markdown and return its path."""
    output_path = unique_markdown_path(directory, stem)
    try:
        output_path.write_text(lesson + "\n", encoding="utf-8")
    except OSError as error:
        logger.error("lesson_write_failed path=%s error=%s", output_path, error)
        raise PipelineError(f"Could not write {output_path}: {error}") from error
    logger.info("lesson_written path=%s", output_path)
    return output_path
