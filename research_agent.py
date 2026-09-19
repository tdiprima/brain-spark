"""Entry point: orchestration and terminal display only."""

import logging
import sys
from pathlib import Path

from config import ConfigError, configure_logging, load_config
from ollama_client import OllamaClient, OllamaError
from pipeline import PipelineError, research, suggest_filename, teach, validate_topic, write_lesson
from profile import ProfileError, load_name, save_name, validate_name

logger = logging.getLogger("research_agent")

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_OLLAMA_ERROR = 3
EXIT_IO_ERROR = 4
EXIT_INTERRUPTED = 130

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

BANNER = (
    "╔══════════════════════════════════╗\n"
    "║   🔬 Research Agent Pipeline 🔬  ║\n"
    "╚══════════════════════════════════╝"
)


def show(text: str, color: str = "") -> None:
    print(f"{color}{text}{RESET}" if color else text)


def show_error(text: str) -> None:
    print(f"{RED}Error: {text}{RESET}", file=sys.stderr)


def prompt_valid(label: str, validator) -> str:
    """Ask until the validator accepts the input."""
    while True:
        raw_value = input(label)
        try:
            return validator(raw_value)
        except ValueError as error:
            show(f"  {error}. Try again.", YELLOW)


def get_user_name(profile_path: str) -> str:
    saved_name = load_name(profile_path)
    if saved_name:
        show(f"\nWelcome back, {saved_name}!", GREEN)
        return saved_name
    name = prompt_valid("\nHi! What's your name? ", validate_name)
    save_name(profile_path, name)
    show(f"Nice to meet you, {name}!", GREEN)
    return name


def run_lesson(client: OllamaClient, topic: str) -> Path:
    show("\n  Researching...", CYAN)
    facts = research(client, topic)
    show("  Teaching...", CYAN)
    lesson = teach(client, topic, facts)
    stem = suggest_filename(client, topic)
    return write_lesson(Path.cwd(), stem, lesson)


def main() -> int:
    try:
        config = load_config()
    except ConfigError as error:
        show_error(str(error))
        return EXIT_CONFIG_ERROR
    configure_logging(config.log_level)

    show(BANNER, CYAN)
    try:
        name = get_user_name(config.profile_path)
        show(f"\nAlright {name}, what do you want to learn about?")
        topic = prompt_valid("  Topic or question: ", validate_topic)
        client = OllamaClient(config.ollama_url, config.ollama_model, config.request_timeout_seconds)
        output_path = run_lesson(client, topic)
    except (KeyboardInterrupt, EOFError):
        show("\nBye!", YELLOW)
        return EXIT_INTERRUPTED
    except OllamaError as error:
        show_error(str(error))
        return EXIT_OLLAMA_ERROR
    except (ProfileError, PipelineError) as error:
        show_error(str(error))
        return EXIT_IO_ERROR

    show(f"\n  Written to: {output_path.name}", CYAN)
    show("\n  Done! :)", GREEN)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
