## How it works:
- First run -> asks name, saves to `user_profile.json`. Next runs -> greets by name.
- Asks topic -> two Ollama calls with capped tokens:
a. Research (800 tok max) - concise facts
b. Teach (1200 tok max) - 10th-grade teacher presentation with real-life ties, comparisons
- Separate call generates short filename
- Writes markdown, prints green "Done! :)"

## Context window strategy:
Each call has explicit `num_predict` cap + prompt instructs word limits. Research stays under 400 words, teaching under 600. No huge blob passed around -> no mid-sentence cutoff.

## Run:
```sh
python3 research_agent.py
```

**Needs:** requests (`pip install requests`), Ollama running with `gemma4:latest` pulled.

**Note:** Change MODEL var if needed.

## Check model

```sh
curl http://localhost:11434/api/generate -d '{"model":"gemma4:latest","prompt":"hello","stream":false}'
```

<br>
