# brain-spark

An interactive CLI that researches any topic using a local LLM, then explains it back to you like an enthusiastic 10th-grade teacher.

## Reading Wikipedia at midnight isn't learning

You look something up. You get a wall of jargon, dry facts, and zero context for why it matters. Ten minutes later you've retained nothing. The information was technically correct but pedagogically useless — no connections to things you already know, no "oh wow" moment, no reason to care.

## A teacher in your terminal

Brain-spark runs a two-stage AI pipeline locally on your machine:

1. **Research** — queries Ollama for concise, factual information on your topic
2. **Teach** — a second pass transforms those facts into a lesson: real-life analogies, comparisons to similar concepts, and a tone that makes you want to keep reading

The output is saved as a clean markdown file with an auto-generated name. Token limits at each stage prevent context window overflow, so lessons never cut off mid-sentence.

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

The resulting `stellar-collapse.md` contains a markdown lesson covering formation mechanics, real-world scale comparisons, contrast with neutron stars, and a closing mind-blowing fact.

## Usage

**Prerequisites**

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- The `gemma4:latest` model pulled (`ollama pull gemma4:latest`)

**Install and run**

```bash
pip install requests
python research_agent.py
```

On first run, it asks your name and saves it to `user_profile.json`. Every run after that, it greets you by name and jumps straight to the topic prompt.

**Configuration**

Edit the constants at the top of `research_agent.py` to swap models or point to a remote Ollama instance:

```python
MODEL = "gemma4:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
```
