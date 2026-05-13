# Placement Experiment: Antigravity Prompt v3
​
Build a Python project that runs a controlled experiment measuring how instruction placement across different **prompt slots** (system message, user message, tool description) affects agent behavior in agentic LLM loops.
​
The project supports two LLM providers: **Ollama** (local, primary) and **Anthropic Claude** (cloud, validation). The provider is selected via environment variable.
​
---
​
## Goal
​
Measure how placing the same instruction in different structural slots affects:
​
1. **Compliance rate** — did the agent follow the instruction in its final text response?
2. **Completion rate** — did the agent finish within the turn cap?
3. **Turns to completion** — how many LLM calls did it take?
4. **Total token usage** — cost proxy
5. **Prefill duration** — turn 1 vs subsequent turns (Ollama only; proxy for KV-cache effectiveness)
​
The instruction under test: **"End your final answer with the marker [DONE]"**
​
Compliance check: case-insensitive search for `[DONE]` in the last 80 characters of the agent's final text response. Use `re.search(r'\[done\]', final_answer[-80:], re.IGNORECASE)`.
​
---
​
## Tech Stack (pin these exact versions)
​
- Python >= 3.10
- `ollama==0.6.2` — official Ollama Python client (released Apr 29, 2026)
- `anthropic==0.99.0` — official Anthropic Python SDK (released May 5, 2026); mark as optional dependency
- `pandas` (latest stable) — CSV handling and analysis
- `matplotlib` (latest stable) — charts
- `python-dotenv` — config
​
Use `uv` for dependency management. Generate `pyproject.toml` with optional dependency group `[anthropic]` so users can install without Anthropic if they only want Ollama. Also generate `uv.lock`.
​
**Do NOT use `litellm`.** Versions 1.82.7 and 1.82.8 had a supply chain incident in March 2026. We use the native SDKs directly instead.
​
---
​
## Repository Structure
​
```
placement-experiment/
├── pyproject.toml
├── README.md
├── .env.example
├── src/placement_experiment/
│   ├── __init__.py
│   ├── config.py               # Loads .env, exposes settings
│   ├── llm_client.py           # Provider abstraction (Ollama + Anthropic)
│   ├── tools.py                # Tool definitions + dispatcher
│   ├── agent_loop.py           # The minimal agent loop
│   ├── placements.py           # The three placement variants
│   ├── runner.py               # Runs trials, appends to CSV, resumable
│   ├── analyze.py              # Reads CSV, prints table, saves chart
│   └── generate_data.py        # Generates 5 sample markdown files
├── data/sample_files/          # Created by generate_data.py
└── results/                    # Created at runtime
    ├── runs.csv
    └── chart.png
```
​
Entry points (all Python CLI, no shell scripts):
- `python -m placement_experiment.generate_data`
- `python -m placement_experiment.runner` (full run; resumable)
- `python -m placement_experiment.runner --smoke-test` (1 run per placement)
- `python -m placement_experiment.analyze`
​
---
​
## The Agent Task
​
Given 5 markdown files in `data/sample_files/`, the agent must:
1. List the files in the directory
2. Read each file
3. Count TODO markers (case-insensitive substring "TODO") in each
4. Produce a final text response summarizing the count per file
​
Generate the 5 files deterministically via `generate_data.py` with known TODO counts: 2, 3, 1, 0, 4. Each file should be 200–400 words of plausible markdown content (project notes, meeting minutes, tech specs — whatever makes sense) with the right number of TODO comments scattered through. Use clear TODO markers like `TODO:` or `<!-- TODO: ... -->` — avoid bare "TODO" that could appear as a substring of another word (e.g., "autodoc"). Filenames: `file_1.md` through `file_5.md`. Make `generate_data.py` idempotent.
​
**Important: all scripts assume CWD is the project root.** The user prompt references `data/sample_files/` as a relative path, and tools resolve it relative to CWD. Document this in the README quickstart.
​
---
​
## Tools (only two)
​
- `list_files(directory: str) -> list[str]`: returns sorted list of file paths in the directory.
- `read_file(path: str) -> str`: returns full file contents.
​
There is **NO** `submit_answer` tool. The agent terminates by returning a text response with no tool calls. Compliance is checked on that final text response. This is intentional — we want to measure placement effects on free-text generation, not on tool argument formatting.
​
### Tool definitions — internal and provider-specific formats
​
Define a **provider-agnostic internal format** that placements.py works with:
​
```python
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema object
```
​
```python
TOOLS = [
    ToolDef(
        name="list_files",
        description="Returns a sorted list of file paths in the given directory.",
        parameters={"type": "object", "properties": {"directory": {"type": "string", "description": "Path to the directory"}}, "required": ["directory"]}
    ),
    ToolDef(
        name="read_file",
        description="Returns the full contents of the file at the given path.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file"}}, "required": ["path"]}
    ),
]
```
​
Placement functions modify `ToolDef.description` (via deep copy). The `llm_client.py` converts `list[ToolDef]` to the provider-specific format before making API calls:
​
**Ollama conversion** (in llm_client.py):
```python
def tooldef_to_ollama(tool: ToolDef) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}
```
​
**Anthropic conversion** (in llm_client.py):
```python
def tooldef_to_anthropic(tool: ToolDef) -> dict:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
```
​
The tool dispatcher in `tools.py` must implement the actual functions (using `os.listdir` / `pathlib` for `list_files`, and file reading for `read_file`). Return results as strings.
​
---
​
## The Three Placement Variants
​
The instruction **"End your final answer with the marker [DONE]"** is placed in exactly one slot per run.
​
### Base values (constant across all variants)
​
```
BASE_SYSTEM = "You are a helpful assistant that uses tools to analyze files."
BASE_USER = "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
INSTRUCTION = "End your final answer with the marker [DONE]"
```
​
### Variant 1: system
​
The instruction is appended to the system message.
​
**What the model sees:**
- System: "You are a helpful assistant that uses tools to analyze files. End your final answer with the marker [DONE]"
- User: "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
- Tools: base definitions (unmodified)
- Model: default (`llama3.1:8b` for Ollama, `claude-sonnet-4-6` for Anthropic)
​
### Variant 2: user
​
The instruction is prepended to the user message.
​
**What the model sees:**
- System: "You are a helpful assistant that uses tools to analyze files."
- User: "End your final answer with the marker [DONE]\n\nFind the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
- Tools: base definitions (unmodified)
- Model: default
​
### Variant 3: tool_description
​
The instruction is appended to the `read_file` tool's description.
​
**What the model sees:**
- System: "You are a helpful assistant that uses tools to analyze files."
- User: "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
- Tools: `list_files` unchanged; `read_file` description becomes "Returns the full contents of the file at the given path. End your final answer with the marker [DONE]"
- Model: default
​
### Implementation in `placements.py`
​
Each placement is a function that returns a provider-agnostic config:
​
```python
@dataclass
class PlacementConfig:
    system_prompt: str
    user_prompt: str
    tools: list[ToolDef]        # Provider-agnostic; llm_client converts to provider format
    model: str
    name: str                   # e.g. "system", "user", "tool_description"
```
​
The agent loop unpacks PlacementConfig into its components and passes them to the llm_client functions.
​
---
​
## LLM Client Abstraction (`llm_client.py`)
​
Two provider implementations behind a common interface. The agent loop calls these functions and never constructs provider-specific message formats directly.
​
### Public interface
​
```python
def chat(messages: list[dict], tools: list[ToolDef], model: str, temperature: float = 0.7, system: str | None = None) -> LLMResponse:
    """Send messages to the LLM. Converts ToolDef to provider-specific format internally.
    Provider is determined by config.LLM_PROVIDER.
    system: For Anthropic, passed as top-level param to messages.create(). For Ollama, ignored (system is in messages array)."""
​
def build_initial_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Construct the initial messages list from a PlacementConfig's system/user prompts.
    For Ollama: [{"role": "system", ...}, {"role": "user", ...}].
    For Anthropic: [{"role": "user", ...}] only (system goes as top-level param in chat())."""
​
def append_tool_interaction(messages: list[dict], response: LLMResponse, tool_calls: list[ToolCall], results: list[str]) -> None:
    """Append the assistant's tool call and tool results to the message history, in-place.
    For Ollama: appends assistant message, then one {"role": "tool"} per result.
    For Anthropic: appends ONE assistant message with all content blocks,
    then ONE {"role": "user"} message with all tool_result blocks.
    This encapsulates the provider-specific message format so the agent loop stays provider-agnostic."""
​
def get_system_prompt_for_api(system_prompt: str) -> str | None:
    """For Anthropic: returns the system_prompt (passed as top-level param to messages.create).
    For Ollama: returns None (system prompt is in the messages array, handled by build_initial_messages)."""
```
​
### Normalized response type
​
```python
@dataclass
class LLMResponse:
    content: str                     # Text content (empty string if only tool calls)
    tool_calls: list[ToolCall]       # List of {name: str, arguments: dict, id: str | None}
    usage: Usage                     # {prompt_tokens: int, completion_tokens: int, total_tokens: int}
    prefill_duration_ms: float | None  # Ollama only; None for Anthropic
    raw_response: Any                # Provider-specific raw response for message history reconstruction.
                                     # Ollama: the response.message object (append directly to history).
                                     # Anthropic: the response.content list of blocks (used in assistant message).
```
​
### Ollama implementation
​
```python
from ollama import chat as ollama_chat
​
# System prompt goes into messages array as {"role": "system", "content": "..."}
# Tools use OLLAMA_TOOLS format (type: function, function: {name, description, parameters})
# Temperature: pass via options dict, NOT as a direct kwarg:
#   ollama_chat(model=model, messages=messages, tools=tools, options={"temperature": 0.7})
#   A direct temperature=0.7 kwarg will be silently ignored.
# Response parsing:
#   - response.message.content → text (can be None if only tool calls; treat as empty string)
#   - response.message.tool_calls → list of tool calls, OR None (NOT empty list) when no tools called.
#     MUST use truthy check: `if response.message.tool_calls:` not `if len(...) > 0:` — the latter crashes on None.
#   - If response has BOTH content AND tool_calls, prioritize tool_calls (dispatch them, continue loop).
#   - response.prompt_eval_count → prompt_tokens (default 0 if missing/None)
#   - response.eval_count → completion_tokens (default 0 if missing/None)
#   - response.prompt_eval_duration → nanoseconds; divide by 1_000_000 for ms (default None if missing)
# Model existence check: on first call, verify model is available via ollama.show(model).
#   If it fails, print: "Model '{model}' not found. Run: ollama pull {model}" and exit with code 1.
```
​
### Anthropic implementation
​
```python
from anthropic import Anthropic
​
# System prompt goes as top-level `system` parameter (NOT in messages array)
# Tools use ANTHROPIC_TOOLS format (name, description, input_schema)
# Response parsing:
#   - response.content is a LIST of blocks
#   - Text blocks: {"type": "text", "text": "..."}
#   - Tool use blocks: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
#   - response.stop_reason: "end_turn" (text response) or "tool_use" (tool call)
#   - response.usage.input_tokens → prompt_tokens
#   - response.usage.output_tokens → completion_tokens
#   - prefill_duration_ms is always None for Anthropic
# Temperature passed directly: temperature=0.7
# max_tokens is required: set to 4096
```
​
### Message history format differences
​
The agent loop maintains a message history. The format of tool calls and tool results in this history differs between providers:
​
**Ollama tool result in history:**
```python
# Append the assistant's raw response to history:
messages.append(response.raw_response)  # The original Ollama message object
# Then append one tool result per call:
{"role": "tool", "content": "<tool result string>"}
```
​
**Anthropic tool call + result in history:**
​
**Critical: Anthropic requires strictly alternating user/assistant messages.** When the model returns multiple `tool_use` blocks in one response, you must:
1. Append ONE assistant message containing ALL the content blocks from the response
2. Dispatch ALL tool calls and collect results
3. Append ONE user message containing ALL `tool_result` blocks together
​
```python
# Step 1: Append the full assistant response (one message, all content blocks):
{"role": "assistant", "content": response.raw_response}  # raw_response stores the Anthropic content blocks list
​
# Step 2: Append ALL tool results as a SINGLE user message:
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
    for tc, result in zip(response.tool_calls, results)
]}
```
​
**Do NOT append separate user messages per tool result** — two user messages in a row causes a 400 error from the Anthropic API.
​
The `llm_client.py` must handle these format differences internally. The agent loop should work with provider-agnostic structures and let the client do the conversion.
​
### Error handling
​
- If Ollama is not running: catch `ConnectionError` / `httpx.ConnectError` and print a clear message: "Ollama is not running at {host}. Start it with `ollama serve`." Exit with code 1.
- If Anthropic API key is missing: print "ANTHROPIC_API_KEY not set. Add it to .env or set the environment variable." Exit with code 1.
- If a tool call references an unknown tool name: return an error string as the tool result (e.g., "Error: unknown tool 'foo'"). Do not crash.
​
---
​
## Agent Loop (`agent_loop.py`)
​
Minimal while-loop. **Truly provider-agnostic** — uses llm_client helper functions for all message formatting.
​
```
1. Initialize turn_count = 0, per_turn_records = []
2. Unpack PlacementConfig: system_prompt, user_prompt, tools, model, name
3. messages = llm_client.build_initial_messages(system_prompt, user_prompt)
4. system_for_api = llm_client.get_system_prompt_for_api(system_prompt)
5. While turn_count < 15:
   a. Try: response = llm_client.chat(messages, tools, model, temperature=0.7, system=system_for_api)
      On exception: return {status: "error", turns: turn_count, final_answer: str(exception), per_turn_records}
   b. Record: {turn: turn_count, prompt_tokens, completion_tokens, prefill_duration_ms}
   c. turn_count += 1
   d. If response.tool_calls (truthy check — can be None or empty):
      - For each tool call: dispatch via tools.dispatch_tool(name, arguments), collect results
      - llm_client.append_tool_interaction(messages, response, tool_calls, results)
      - Note: if response also has text content alongside tool_calls, ignore the text — tool_calls take priority
      - Continue loop
   e. Else (text response, no tool calls):
      - final_answer = response.content
      - status = "success"
      - Break
6. If loop exits without break: status = "timeout", final_answer = None
7. Return: {status, turns: turn_count, final_answer, per_turn_records}
```
​
The agent loop never constructs provider-specific message dicts. All formatting goes through llm_client helpers.
​
If a tool call references an unknown tool name, `dispatch_tool` should return an error string (e.g., "Error: unknown tool 'foo'"). The agent loop includes this as the tool result and continues — it does not crash.
​
Use `temperature=0.7`. Do NOT set a fixed seed — we want trajectory variance across runs.
​
---
​
## Runner (`runner.py`)
​
### Resumable execution
​
On startup, read `results/runs.csv` if it exists. Parse existing `(placement, run_id)` pairs. Skip any already-completed pairs. Append new rows as each run finishes — do not buffer to the end. If the process crashes, partial progress is preserved and the next invocation picks up where it left off.
​
### Run matrix
​
**Ollama runs (LLM_PROVIDER=ollama):**
- Placements: system, user, tool_description
- Runs per placement: 50
- Total: 150 runs
​
**Anthropic runs (LLM_PROVIDER=anthropic):**
- Placements: system, user, tool_description
- Runs per placement: 20
- Total: 60 runs
​
### Per-run logic
​
```
For each applicable placement:
  For run_id in 0..(N-1):
    Skip if (placement, run_id) already in CSV.
    Construct PlacementConfig.
    Try:
      Execute agent_loop.
    Except unexpected exception:
      Set status="error", final_answer=str(exception), turns=0, per_turn_records=[]
    Compute:
      - compliance: True/False if status == "success"; None if status != "success"
      - prefill_ms_turn_1: per_turn_records[0].prefill_duration_ms (None for Anthropic or if no turns)
      - prefill_ms_subsequent_mean: mean of per_turn_records[1:].prefill_duration_ms; None if <=1 turn or Anthropic
    Append row to CSV.
```
​
### Model checks and Warm-up on startup
​
Before starting runs, verify required models are available:
- For Ollama: call `ollama.show(OLLAMA_MODEL)`. If it fails: print "Model '{model}' not found. Run: ollama pull {model}" and exit.
  - **Warm-up:** If Ollama is available, make a single throwaway chat request (e.g., sending "hi" to the model) and discard the response before starting the experiment loop. This forces the model into memory and prevents the first run's prefill duration from being artificially inflated by a "cold start" if resuming the script.
- For Anthropic: verify `ANTHROPIC_API_KEY` is set. If not: print clear error and exit.
​
### CSV columns
​
```
provider, placement, run_id, status, turns, total_tokens, prefill_ms_turn_1, prefill_ms_subsequent_mean, compliance, final_answer
```
​
- `provider`: "ollama" or "anthropic"
- `status`: "success", "timeout", or "error"
- `compliance`: True, False, or empty (for non-success runs)
- `final_answer`: truncated to 500 chars, newlines replaced with spaces
​
**Use `csv.DictWriter` with `quoting=csv.QUOTE_ALL`** for writing rows. The `final_answer` field will contain commas and quotes that break naive CSV writing.
​
### Progress logging
​
Print progress every 10 runs. Include running compliance rate per placement. After the first 10 runs, print an ETA for remaining runs.
​
### Smoke test
​
`--smoke-test` flag: 1 run per applicable placement (3 for both Ollama and Anthropic). Verify CSV output is valid and all columns are populated.
​
After the smoke test completes, check if ALL successful runs have `compliance=False`. If so, print a warning: "Warning: No runs were compliant. The model may not follow format instructions reliably. Consider trying a larger model or a different instruction before running the full experiment."
​
### Keep-alive note
​
When running Ollama, recommend setting `OLLAMA_KEEP_ALIVE=30m` to prevent model unloading between runs. Document this in the README. If the model unloads mid-experiment, turn-1 prefill times will spike erratically and contaminate the cache signal.
​
---
​
## Analysis (`analyze.py`)
​
Reads `results/runs.csv`. Groups by `(provider, placement)`.
​
### Per group, compute:
​
- **Completion rate**: success / total, with Wilson 95% CI
- **Compliance rate** (among completions only): compliance==True / status==success, with Wilson 95% CI
- **Mean turns** to completion (success only), with standard error
- **Mean total_tokens** (success only), with standard error
- **Mean prefill_ms_turn_1** (Ollama success only), with standard error
- **Mean prefill_ms_subsequent_mean** (Ollama success only, runs with >1 turn), with standard error
​
**Handle zero-completion groups gracefully.** If a placement has zero successful completions, report "N/A" for compliance rate, mean turns, and mean tokens — do not crash with a division by zero.
​
### Output
​
1. Print a summary table to stdout using `pandas.DataFrame.to_string()`.
2. Save a chart to `results/chart.png`:
   - If Ollama-only data: 2×3 grid of bar charts (completion_rate, compliance_rate, mean_turns, mean_tokens, prefill_turn_1, prefill_subsequent)
   - If both providers present: group bars by provider within each chart for direct comparison (but omit prefill charts for Anthropic since those values are null)
   - X-axis: placement name
   - Error bars: Wilson CI for rates, standard error for means
   - Use matplotlib defaults — labels, titles, no fancy styling
​
---
​
## Configuration (`.env.example`)
​
```
# Provider: "ollama" or "anthropic"
LLM_PROVIDER=ollama
​
# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_KEEP_ALIVE=30m
​
# Anthropic settings (only needed when LLM_PROVIDER=anthropic)
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6
```
​
---
​
---
​
## README Requirements
​
- One-paragraph description: what this project is and why.
- Quickstart for Ollama:
  1. Install Ollama; ensure it's running
  2. `ollama pull llama3.1:8b`
  3. `uv sync`
  4. `python -m placement_experiment.generate_data`
  5. `python -m placement_experiment.runner --smoke-test`
  6. `python -m placement_experiment.runner`
  7. `python -m placement_experiment.analyze`
- How to run validation on Anthropic:
  1. `uv sync --extra anthropic`
  2. Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`
  3. `python -m placement_experiment.runner` (runs 60 validation trials)
  4. `python -m placement_experiment.analyze` (now shows both providers)
- Note: runner is resumable — safe to interrupt and re-run.
- Note: set `OLLAMA_KEEP_ALIVE=30m` for stable prefill measurements.
- Caveats:
  - Results are model- and task-specific.
  - This experiment measures **slot effects**, not text-position effects. Each slot (system message, user message, tool description) has its own mechanics built into how the model was trained.
  - Direction of effects is more transferable than magnitude. Open-weight models typically show larger placement sensitivity than frontier models.
​
---
​
## Verification Before Completion
​
Before declaring the project done, verify by:
​
1. `uv sync` runs cleanly.
2. `python -m placement_experiment.generate_data` produces 5 files with correct TODO counts (verify programmatically: read each file, count "TODO" occurrences case-insensitively, assert counts are [2, 3, 1, 0, 4]).
3. `python -m placement_experiment.runner --smoke-test` with `LLM_PROVIDER=ollama` completes 3 runs (one per placement) and writes 3 rows to CSV. Verify each row has: status filled, turns > 0, total_tokens > 0, prefill_ms_turn_1 > 0 where status == success.
4. `python -m placement_experiment.analyze` reads the smoke-test CSV and produces both a table on stdout and a chart at `results/chart.png`. With only 3 rows the stats will be meaningless — that's fine, we're checking the pipeline.
​
**Do NOT run the full 150-run or 60-run experiment** — that's for the user to do on their own machine.
​
If Ollama is not available in the build environment, document the steps that require it and verify all non-Ollama code paths (imports, config loading, analysis on a manually-created sample CSV).
​
---
​
## Python Best Practices
​
Follow these throughout all modules:
​
- **Type hints** on all function signatures and return types. Use `from __future__ import annotations` at the top of every module for cleaner syntax.
- **Docstrings** on all public functions, classes, and modules. One-line for simple functions, multi-line (Google style) for complex ones.
- **`logging` module** for all progress output, warnings, and errors — not bare `print()`. Configure with a simple `logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")` in the entry points.
- **`pathlib.Path`** for all file system operations — not `os.path`. Use `Path.resolve()` for safe path handling.
- **`if __name__ == "__main__"`** guard in every module that serves as an entry point (`runner.py`, `analyze.py`, `generate_data.py`).
- **`argparse`** for CLI arguments (needed for `--smoke-test`).
- **Constants** in `UPPER_CASE` at module top (e.g., `BASE_SYSTEM`, `INSTRUCTION`, `MAX_TURNS = 15`).
- **Dataclasses** for all structured data (PlacementConfig, LLMResponse, ToolCall, Usage, ToolDef — already specified). No mutable default arguments — use `field(default_factory=list)`.
- **Deep copy** when modifying tools for the `tool_description` placement — never mutate the shared `TOOLS` list. Use `copy.deepcopy()`.
- **Clean imports** — no wildcard imports, group into stdlib / third-party / local with blank lines between groups.
- **Ruff** for linting and formatting. Add to `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 120
  target-version = "py310"
  ```
- **Specific exceptions** — catch `httpx.ConnectError` for Ollama connection failures, `anthropic.APIError` for Anthropic failures. Avoid bare `except Exception`.
​
---
​
## What NOT To Do
​
- **No agent frameworks** (LangChain, LangGraph, Pydantic AI, CrewAI, etc.). The loop is hand-written and fully visible in `agent_loop.py`.
- **No `litellm`**. Use native SDKs only (`ollama` and `anthropic`).
- **No async/concurrent execution**. Sequential only — simpler to reason about.
- **No web UI**. CLI only.
- **No caching layers**. Measuring real prefill behavior is the point.
- **No observability frameworks** (OpenTelemetry, etc.). Metrics are collected inline.
- **No committed `.env` files or API keys** — only `.env.example`.
​
---
​
## Deliverables
​
When complete, produce:
1. A task list of what was built.
2. A brief README walkthrough.
3. The smoke-test CSV (3 rows) and the smoke-test chart.
4. Note any deviations from this spec and why.