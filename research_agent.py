#!/usr/bin/env python3
"""Interactive research agent - queries Ollama, teaches like a 10th-grade teacher."""

import json
import os
import re
import sys
import requests

PROFILE_FILE = "user_profile.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:latest"

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colored(text, color):
    return f"{color}{text}{RESET}"


def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return None


def save_profile(profile):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)


def greet_user():
    profile = load_profile()
    if profile and profile.get("name"):
        name = profile["name"]
        print(colored(f"\nWelcome back, {name}!", CYAN + BOLD))
    else:
        print(colored("\nHey there! First time here.", CYAN + BOLD))
        name = input(colored("What's your name? ", YELLOW)).strip()
        if not name:
            name = "Friend"
        save_profile({"name": name})
        print(colored(f"\nNice to meet you, {name}! Saved for next time.\n", GREEN))
    return name


def query_ollama(prompt, max_tokens=1500):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError:
        print(colored("\nERROR: Can't reach Ollama. Is it running?", "\033[91m"))
        sys.exit(1)
    except requests.Timeout:
        print(colored("\nERROR: Ollama timed out.", "\033[91m"))
        sys.exit(1)
    except requests.HTTPError as err:
        print(colored(f"\nERROR: Ollama returned {err}", "\033[91m"))
        sys.exit(1)


def research_topic(topic):
    prompt = (
        f"Research this topic concisely (under 400 words, factual, key points only): {topic}"
    )
    print(colored("\n  Researching...", MAGENTA))
    return query_ollama(prompt, max_tokens=800)


def teach_topic(topic, raw_info):
    prompt = f"""You are an enthusiastic 10th-grade teacher. A student asked about: "{topic}"

Here are the key facts:
{raw_info}

Now teach this to the student. Requirements:
- Make it sound genuinely interesting and exciting
- Relate key points to real life (everyday examples they'd recognize)
- Compare/contrast with something similar so the concept clicks
- Use a conversational, encouraging tone
- Keep it under 800 words
- Use markdown formatting (headers, bold, bullet points)
- End with a "mind-blowing" takeaway or fun fact

Do NOT use a title/heading at the very top (I'll add one). Jump straight into teaching."""

    print(colored("  Teaching...", MAGENTA))
    return query_ollama(prompt, max_tokens=1600)


def generate_filename(topic):
    prompt = (
        f"Generate a short, punchy filename (2-4 words, kebab-case, no extension) "
        f"for a research note about: {topic}\n"
        f"Reply with ONLY the filename, nothing else."
    )
    raw = query_ollama(prompt, max_tokens=30)
    slug = re.sub(r"[^a-z0-9\-]", "", raw.lower().strip().replace(" ", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug or len(slug) < 3:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()[:40]).strip("-")
    return f"{slug}.md"


def write_markdown(filename, topic, content):
    md = f"# {topic}\n\n{content}\n"
    with open(filename, "w") as f:
        f.write(md)


def main():
    print(colored("\n╔══════════════════════════════════╗", CYAN))
    print(colored("║   🔬 Research Agent Pipeline 🔬  ║", CYAN + BOLD))
    print(colored("╚══════════════════════════════════╝", CYAN))

    name = greet_user()

    print(colored(f"\nAlright {name}, what do you want to learn about?", YELLOW + BOLD))
    topic = input(colored("  Topic or question: ", YELLOW)).strip()

    if not topic:
        print(colored("No topic given. Bye!", "\033[91m"))
        sys.exit(0)

    raw_info = research_topic(topic)
    lesson = teach_topic(topic, raw_info)
    filename = generate_filename(topic)

    write_markdown(filename, topic, lesson)

    print(colored(f"\n  Written to: {filename}", CYAN))
    print(colored("\n  Done! :)\n", GREEN + BOLD))


if __name__ == "__main__":
    main()
