# brain-spark 🧠 ✨

An interactive CLI that researches a topic using OpenAI's `gpt-5.2`, then explains it back to you like an enthusiastic professor.

## Reading Wikipedia at midnight isn't learning

You look something up. You get a wall of jargon, dry facts, and zero context for why it matters. Ten minutes later you've retained nothing. The information was technically correct but pedagogically useless — no connections to things you already know, no "oh wow" moment, no reason to care.

## A teacher in your terminal

Brain-spark runs a two-stage AI pipeline using the OpenAI Responses API:

1. **Research** — asks GPT-5.2 for concise, factual information on your topic
2. **Teach** — a second pass transforms those facts into a lesson with real-life analogies and comparisons

A final request suggests a filename. The lesson is saved as a markdown file in the current directory without overwriting existing files. If filename generation fails, the topic supplies the filename instead.

Each stage has an output token budget. Research and teaching retry once with a larger budget if generation is cut off; if the retry is also incomplete, the application reports an error instead of saving a partial lesson.

Generation sends your topic and generated facts to OpenAI and uses your API account's billing. The requests use `store: false`. Research draws on the model's knowledge; it does not browse the web.

It remembers your name between sessions and greets you personally. Colored terminal output keeps the interaction visually clear.

## Example

```
╔══════════════════════════════════╗
║   🔬 Research Agent Pipeline 🔬  ║
╚══════════════════════════════════╝

Welcome back, Alex!

Alright Alex, what do you want to learn about?
  Topic or question: How do black holes form?

  Researching...
  Teaching...

  Written to: stellar-collapse.md

  Done! :)
```

## Usage

**Prerequisites**

- Python 3.13+
- An OpenAI API account with access to `gpt-5.2`
- `OPENAI_API_KEY` exported in your shell environment

**Install and run**

```bash
pip install requests
# Only if OPENAI_API_KEY is not already set:
export OPENAI_API_KEY="your-api-key"
python research_agent.py
```

The key is read from the environment at startup; a missing or blank key produces a configuration error before any interactive prompts. The application does not load `.env` files automatically.

On first run, it asks your name and saves it to `~/.config/brain-spark/user_profile.json` (or under `XDG_CONFIG_HOME` when set). Every run after that, it greets you by name and jumps straight to the topic prompt. Your name is kept in the local profile and is not included in model prompts.

**Configuration**

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | Required | API key used for bearer authentication |
| `OPENAI_MODEL` | `gpt-5.2` | Responses API model supporting `reasoning.effort: none` |
| `OPENAI_TIMEOUT_SECONDS` | `300` | Timeout in seconds for each HTTP request |
| `BRAIN_SPARK_PROFILE` | `~/.config/brain-spark/user_profile.json` | Path to saved profile; respects `XDG_CONFIG_HOME` by default |
| `BRAIN_SPARK_LOG_LEVEL` | `WARNING` | Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL |

Requests go to `https://api.openai.com/v1/responses`.

## Project structure

```
research_agent.py   # Entry point — orchestration and display
pipeline.py         # Research → teach → filename → write pipeline
openai_client.py    # OpenAI Responses HTTP transport with typed exceptions
user_profile.py     # User profile storage and retrieval
config.py           # Configuration from environment variables
sanitize.py         # Terminal control-character escaping
```

## Testing

```sh
pip install requests pytest
python3 -m pytest
```

Tests use fake keys and mocked HTTP responses, with temporary profile/output directories. They do not call OpenAI or change your shell's environment variables.

API references: [text generation](https://developers.openai.com/api/docs/guides/text), [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2).
