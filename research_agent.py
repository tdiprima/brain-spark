#!/usr/bin/env python3
"""Interactive research agent - queries Ollama, teaches like a 10th-grade teacher."""

import sys

import profile as user_profile
from ollama_client import OllamaConnectionError, OllamaHTTPError, OllamaTimeoutError
from pipeline import generate_filename, research_topic, teach_topic, write_markdown

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"


def colored(text, color):
    return f"{color}{text}{RESET}"


def greet_user():
    name = user_profile.get_name()
    if name:
        print(colored(f"\nWelcome back, {name}!", CYAN + BOLD))
    else:
        print(colored("\nHey there! First time here.", CYAN + BOLD))
        name = input(colored("What's your name? ", YELLOW)).strip()
        if not name:
            name = "Friend"
        user_profile.save_profile(name)
        print(colored(f"\nNice to meet you, {name}! Saved for next time.\n", GREEN))
    return name


def run_pipeline(topic):
    """Research → teach → filename → write. Returns output filename."""
    print(colored("\n  Researching...", MAGENTA))
    raw_info = research_topic(topic)
    if not raw_info:
        raise ValueError("Research step returned empty. Check Ollama/model.")
    print(colored(f"  Got {len(raw_info.split())} words of research.", GREEN))

    print(colored("  Teaching...", MAGENTA))
    lesson = teach_topic(topic, raw_info)
    if not lesson:
        raise ValueError("Teaching step returned empty. Check Ollama/model.")
    print(colored(f"  Got {len(lesson.split())} words of lesson.", GREEN))

    filename = generate_filename(topic)
    write_markdown(filename, topic, lesson)
    return filename


def main():
    print(colored("\n╔══════════════════════════════════╗", CYAN))
    print(colored("║   🔬 Research Agent Pipeline 🔬  ║", CYAN + BOLD))
    print(colored("╚══════════════════════════════════╝", CYAN))

    name = greet_user()

    print(colored(f"\nAlright {name}, what do you want to learn about?", YELLOW + BOLD))
    topic = input(colored("  Topic or question: ", YELLOW)).strip()

    if not topic:
        print(colored("No topic given. Bye!", RED))
        sys.exit(0)

    try:
        filename = run_pipeline(topic)
    except (OllamaConnectionError, OllamaTimeoutError, OllamaHTTPError, ValueError) as err:
        print(colored(f"\n  ERROR: {err}", RED))
        sys.exit(1)

    print(colored(f"\n  Written to: {filename}", CYAN))
    print(colored("\n  Done! :)\n", GREEN + BOLD))


if __name__ == "__main__":
    main()
