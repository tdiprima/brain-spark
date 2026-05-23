import re
from ollama_client import query_ollama


def research_topic(topic):
    """Return bullet-point facts about topic from Ollama."""
    prompt = (
        f"List the key facts about this topic in bullet points. No introduction, "
        f"no preamble, just the facts. Under 500 words: {topic}"
    )
    return query_ollama(prompt, max_tokens=800)


def teach_topic(topic, raw_info):
    """Transform raw facts into a 10th-grade-teacher lesson."""
    prompt = f"""You are an enthusiastic 10th-grade teacher. A student asked about: "{topic}"

Here are the key facts:
{raw_info}

Now teach this to the student. Requirements:
- Make it easy to memorize
- Make it sound genuinely interesting and exciting
- Relate key points to real life (everyday examples they'd recognize)
- Compare/contrast with something similar so the concept clicks
- Use a conversational, encouraging tone
- Keep it under 500 words
- Use markdown formatting (headers, bold, bullet points)
- End with a "mind-blowing" takeaway or fun fact

Do NOT use a title/heading at the very top (I'll add one). Jump straight into teaching."""
    return query_ollama(prompt, max_tokens=1200)


def generate_filename(topic):
    """Ask Ollama for a kebab-case filename; fall back to topic slug."""
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
    """Write lesson to a markdown file."""
    md = f"# {topic}\n\n{content}\n"
    with open(filename, "w") as f:
        f.write(md)
