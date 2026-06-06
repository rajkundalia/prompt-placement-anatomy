# Placement Experiment — Sequential Prompts for Claude Code

Use the full spec in `antigravity-prompt-v3.md` as the reference. These four prompts are the execution steps — paste them into Claude Code one at a time, verify each stage works before moving to the next.

---

## ERRATA — Append these notes when pasting each prompt

### For all prompts:

> **Python best practices — follow throughout:**
> - `from __future__ import annotations` at the top of every module
> - Type hints on all function signatures and return types
> - Docstrings on all public functions and classes (Google style)
> - Use `logging` module for all output (not bare `print()`). Configure with `logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")` in entry points.
> - Use `pathlib.Path` for all file operations, not `os.path`
> - `if __name__ == "__main__"` guard in all entry-point modules
> - `argparse` for CLI arguments
> - Constants in `UPPER_CASE`. No mutable default arguments in dataclasses — use `field(default_factory=list)`.
> - Add ruff config to pyproject.toml: `[tool.ruff]` with `line-length = 120` and `target-version = "py310"`
> - Catch specific exceptions (`httpx.ConnectError`, `anthropic.APIError`), not bare `except Exception`
> - Use `copy.deepcopy()` when modifying tool definitions for placements — never mutate shared objects
> - Clean imports: group into stdlib / third-party / local with blank lines between groups
>
> **CWD assumption:** All scripts assume CWD is the project root. The user prompt references `data/sample_files/` as a relative path. Document this in the README.

These are gaps in the prompts below. Copy the relevant notes and paste them at the end of each prompt before submitting to Claude Code.

### For Prompt 1 (agent loop):

> **Ollama temperature:** Pass temperature via `options={"temperature": 0.7}`, NOT as a direct keyword argument. Example: `ollama.chat(model=model, messages=messages, tools=tools, options={"temperature": 0.7})`. A direct `temperature=0.7` kwarg will be silently ignored and the model will use its default temperature.
>
> **Null tool_calls:** Ollama's `response.message.tool_calls` can be `None` (not empty list) when the model returns a text response. Use a truthy check (`if response.message.tool_calls:`) not a length check (`if len(response.message.tool_calls) > 0:`) — the latter crashes on None.
>
> **Tool calls + text simultaneously:** If the response contains both `content` (text) AND `tool_calls`, prioritize tool_calls — dispatch them and continue the loop. The text is the model "thinking out loud" before calling tools.
>
> **Model existence check:** On startup, verify the model is available by calling `ollama.show(model)`. If it raises an error, print: "Model '{model}' not found. Run: ollama pull {model}" and exit with code 1.
>
> **API error handling:** Wrap the `ollama.chat()` call in a try/except. If the API call fails (connection error, model error, etc.), the agent loop should return `status="error"` with the error message, not crash.

### For Prompt 2 (runner):

> **CSV escaping:** Use `csv.DictWriter` with `quoting=csv.QUOTE_ALL` for writing rows. The `final_answer` field will contain commas and quotes that break naive CSV writing.
>
> **Error handling:** Wrap each run in try/except. If the agent loop returns `status="error"` or raises an unexpected exception, write the row with `status="error"`, `compliance=None`, and continue to the next run. Do not crash the experiment.
>
> **Smoke test sanity check:** After the smoke test completes, if ALL runs show `compliance=False`, print a warning: "No runs were compliant. The model may not follow format instructions reliably. Consider trying a larger model or a different instruction." This saves the user from running 150 trials that produce no useful signal.

### For Prompt 3 (analysis):

> **Division by zero:** If a placement has zero successful completions, the compliance rate computation divides by zero. Handle this gracefully — report "N/A" for that placement's compliance rate, not crash.

### For Prompt 4 (Anthropic):

> **Anthropic tool conversion:** Prompt 1 already uses the provider-agnostic `ToolDef` dataclass with `tooldef_to_ollama()`. For Prompt 4, add the corresponding Anthropic conversion:
> ```python
> def tooldef_to_anthropic(tool: ToolDef) -> dict:
>     return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
> ```
> No refactoring of existing code needed — just add the new conversion function and use it in the Anthropic chat path.
>
> **Critical — multiple tool results must be ONE message:** Anthropic requires strictly alternating user/assistant messages. When the model returns multiple `tool_use` blocks in one response:
> 1. Append ONE assistant message containing ALL the content blocks from the response
> 2. Dispatch ALL tool calls and collect results
> 3. Append ONE user message containing ALL `tool_result` blocks:
>    `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id1", "content": "result1"}, {"type": "tool_result", "tool_use_id": "id2", "content": "result2"}]}`
>
> Do NOT append separate user messages per tool result — Anthropic will reject the request with a 400 error about message ordering.

---

## Prompt 1: Project setup + Ollama client + tools + agent loop

```
Create a Python project called `placement-experiment` with `uv` and `pyproject.toml`.

Dependencies: ollama==0.6.2, pandas, matplotlib, python-dotenv.
Python >= 3.10.

Structure:
  src/placement_experiment/
    __init__.py
    config.py
    llm_client.py
    tools.py
    agent_loop.py
    generate_data.py
  data/sample_files/
  results/
  .env.example

Step 1 — config.py:
Load .env with python-dotenv. Expose: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL (default llama3.1:8b), LLM_PROVIDER (default ollama).

Step 2 — generate_data.py:
Generate 5 markdown files in data/sample_files/ (file_1.md through file_5.md). Each should be 200-400 words of plausible project notes with TODO comments. Known TODO counts: file_1=2, file_2=3, file_3=1, file_4=0, file_5=4. Idempotent — overwrites on re-run. Add a verification step that reads each file back and asserts the TODO count.

Run as: python -m placement_experiment.generate_data

Step 3 — tools.py:
Two tools only:
- list_files(directory: str) -> list[str]: returns sorted file paths using pathlib
- read_file(path: str) -> str: returns file contents

Define a provider-agnostic ToolDef dataclass:
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema

TOOLS = [
    ToolDef(name="list_files", description="Returns a sorted list of file paths in the given directory.", parameters={"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}),
    ToolDef(name="read_file", description="Returns the full contents of the file at the given path.", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
]

Include a dispatch function: dispatch_tool(name: str, arguments: dict) -> str that calls the right function and returns the result as a string. If tool name is unknown, return "Error: unknown tool '{name}'" instead of crashing.

Step 4 — llm_client.py (Ollama only for now):

Conversion function:
def tooldef_to_ollama(tool: ToolDef) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}

A chat() function that converts ToolDefs to Ollama format, calls ollama.chat(), and returns a normalized dataclass:

@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    usage: Usage
    prefill_duration_ms: float | None
    raw_response: Any  # Ollama: response.message object. Anthropic: response.content blocks list.
                       # Used by append_tool_interaction to reconstruct assistant message for history.

Signature: chat(messages: list[dict], tools: list[ToolDef], model: str, temperature: float = 0.7) -> LLMResponse

Internally: convert tools via tooldef_to_ollama, pass temperature via options={"temperature": temperature} (NOT as a direct kwarg — a direct kwarg is silently ignored by Ollama).

Parse Ollama's response:
- content from response.message.content (empty string if None)
- tool_calls from response.message.tool_calls — WARNING: can be None, not empty list. Use truthy check.
- Each tool_call has .function.name and .function.arguments
- prompt_tokens from response.prompt_eval_count (default 0 if missing)
- completion_tokens from response.eval_count (default 0 if missing)
- prefill_duration_ms from response.prompt_eval_duration / 1_000_000 (default None if missing)

Also define these helper functions:

def build_initial_messages(system_prompt: str | None, user_prompt: str) -> list[dict]:
    """Build the initial messages list. If system_prompt is None, omit system message."""
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

def append_tool_interaction(messages: list[dict], response: LLMResponse, tool_calls: list[ToolCall], results: list[str]) -> None:
    """Append assistant tool call and tool results to message history, in-place.
    For Ollama: append assistant message, then one {"role": "tool"} per result."""
    messages.append(response.raw_response)  # The original Ollama message object
    for tc, result in zip(tool_calls, results):
        messages.append({"role": "tool", "content": result})

If Ollama is not reachable, catch httpx.ConnectError and print: "Ollama is not running at {host}. Start it with `ollama serve`." Then exit with code 1.
On first call, verify model exists via ollama.show(model). If not found: "Model '{model}' not found. Run: ollama pull {model}" and exit.

Step 5 — agent_loop.py:
A run_agent(system_prompt, user_prompt, tools, model, temperature=0.7) function.

Logic:
- messages = llm_client.build_initial_messages(system_prompt, user_prompt)
- While turn_count < 15:
  - Try: response = llm_client.chat(messages, tools, model, temperature)
    On exception: return {status: "error", turns: turn_count, final_answer: str(exception), per_turn_records: per_turn_records}
  - Record turn data: {turn, prompt_tokens, completion_tokens, prefill_duration_ms}
  - turn_count += 1
  - If response.tool_calls (truthy check):
    - For each tool call: dispatch via tools.dispatch_tool, collect results
    - llm_client.append_tool_interaction(messages, response, response.tool_calls, results)
    - If response also has text content alongside tool_calls, ignore the text — tool_calls take priority
    - Continue
  - Else:
    - final_answer = response.content
    - status = "success"
    - break
- If loop exits without break: status = "timeout", final_answer = None
- Return a dict: {status, turns, final_answer, per_turn_records}

Follow Python best practices throughout: type hints on all functions, docstrings, use logging module (not print), pathlib.Path for file operations, from __future__ import annotations at the top of every module.

Step 6 — Verify:
Create a small test script or __main__ block that:
1. Generates the sample data
2. Runs a single agent invocation with system="You are a helpful assistant that uses tools to analyze files." and user="Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
3. Prints the final answer, status, and turn count

This must produce a successful completion with the correct TODO counts before proceeding.
```

**Verify before moving on:** Run the test. The agent should complete successfully, call list_files once, read_file five times, and return a summary with the correct counts (2, 3, 1, 0, 4). If it doesn't, debug until it does.

---

## Prompt 2: Placements + runner

```
I have a working agent loop in placement-experiment/. The agent can list files, read them, and produce a final text answer using Ollama.

Now add placement variants and the experiment runner.

Step 1 — placements.py:

Define a dataclass:
@dataclass
class PlacementConfig:
    name: str               # "system", "user", "tool_description"
    system_prompt: str
    user_prompt: str
    tools: list[ToolDef]    # From tools.py; provider-agnostic, llm_client converts
    model: str

Import TOOLS from tools.py (the list[ToolDef]).

Constants:
BASE_SYSTEM = "You are a helpful assistant that uses tools to analyze files."
BASE_USER = "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
INSTRUCTION = "End your final answer with the marker [DONE]"

Four functions, each returning a PlacementConfig:

1. system_placement():
   - system_prompt = f"{BASE_SYSTEM} {INSTRUCTION}"
   - user_prompt = BASE_USER
   - tools = TOOLS (unmodified)
   - model = config.OLLAMA_MODEL

2. user_placement():
   - system_prompt = BASE_SYSTEM
   - user_prompt = f"{INSTRUCTION}\n\n{BASE_USER}"
   - tools = TOOLS (unmodified)
   - model = config.OLLAMA_MODEL

3. tool_description_placement():
   - system_prompt = BASE_SYSTEM
   - user_prompt = BASE_USER
   - tools = deep copy of TOOLS with read_file's .description changed to: f"{original_description} {INSTRUCTION}"
   - model = config.OLLAMA_MODEL

Step 2 — runner.py:

Resumable runner.

On startup: read results/runs.csv if it exists, collect set of (placement, run_id) tuples already done.

For each placement in [system, user, tool_description]:
  For run_id in 0..49:
    Skip if already done.
    Get PlacementConfig from placement function.
    Run agent_loop.run_agent(config.system_prompt, config.user_prompt, config.tools, config.model).
    Compute:
      - compliance: if status == "success", check re.search(r'\[done\]', final_answer[-80:], re.IGNORECASE). True/False. If status != "success", compliance = None.
      - prefill_ms_turn_1 = per_turn_records[0]["prefill_duration_ms"]
      - prefill_ms_subsequent_mean = mean of per_turn_records[1:] prefill_duration_ms values; None if <= 1 turn
    Append row to CSV with columns: provider, placement, run_id, status, turns, total_tokens, prefill_ms_turn_1, prefill_ms_subsequent_mean, compliance, final_answer
    - provider = "ollama"
    - final_answer truncated to 500 chars, newlines -> spaces

Create results/ directory if it doesn't exist.
Print progress every 10 runs with running compliance rate.
After first 10 runs, print ETA.

Add --smoke-test flag (via argparse): if set, run 1 trial per placement (3 total) instead of 50.

Run as: python -m placement_experiment.runner
Or: python -m placement_experiment.runner --smoke-test

Step 3 — Verify:
1. Run --smoke-test
2. Confirm results/runs.csv has 3 rows, all columns populated
3. Confirm at least some runs have compliance = True or False (not all None)
4. Interrupt and re-run --smoke-test — it should skip the 3 existing rows and do nothing
```

**Verify before moving on:** Smoke test passes. CSV has 3 valid rows. Re-run skips existing rows.

---

## Prompt 3: Analysis

```
I have a working placement experiment runner that produces results/runs.csv with columns: provider, placement, run_id, status, turns, total_tokens, prefill_ms_turn_1, prefill_ms_subsequent_mean, compliance, final_answer.

Now add analysis.

Step 1 — analyze.py:

Read results/runs.csv with pandas.

Group by (provider, placement). For each group compute:
- completion_rate: count(status=="success") / count(total), with Wilson 95% CI
- compliance_rate: count(compliance==True) / count(status=="success"), with Wilson 95% CI
- mean_turns: mean of turns where status=="success", with standard error
- mean_total_tokens: mean of total_tokens where status=="success", with standard error
- mean_prefill_turn_1: mean of prefill_ms_turn_1 where status=="success" and value is not null, with standard error
- mean_prefill_subsequent: mean of prefill_ms_subsequent_mean where status=="success" and value is not null, with standard error

For Wilson confidence interval, implement it or use a simple formula:
  p_hat = successes / n
  z = 1.96
  denominator = 1 + z**2 / n
  center = (p_hat + z**2 / (2*n)) / denominator
  margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
  ci_low = center - margin
  ci_high = center + margin

Print a summary table to stdout with pandas to_string().

Generate results/chart.png:
- 2x3 grid of bar charts: completion_rate, compliance_rate, mean_turns, mean_tokens, prefill_turn_1, prefill_subsequent
- X-axis: placement name
- Error bars where applicable
- If multiple providers present, group bars by provider within each subplot
- Omit prefill charts for anthropic data (values are null)
- matplotlib defaults, clear labels and title

Run as: python -m placement_experiment.analyze

Step 2 — Verify:
Run analyze on the smoke-test CSV. It should print a table (stats will be meaningless with 3 rows — that's fine) and save a chart. Open the chart and confirm it renders correctly.
```

**Verify before moving on:** Table prints. Chart renders. Both look structurally correct even if the numbers are from only 3 runs.

---

## Prompt 4: Anthropic provider support

```
I have a working placement experiment with Ollama: agent loop, placements, runner, analysis. Now add Anthropic Claude as a second provider for validation runs.

Add anthropic==0.99.0 as an optional dependency in pyproject.toml under [project.optional-dependencies] anthropic = ["anthropic>=0.99.0"].

Step 1 — Update llm_client.py:

Add an Anthropic implementation alongside the existing Ollama one. Select based on config.LLM_PROVIDER.

Add ToolDef-to-Anthropic conversion:
def tooldef_to_anthropic(tool: ToolDef) -> dict:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}

Anthropic chat() implementation:
- Use: from anthropic import Anthropic; client = Anthropic()
- System prompt is a TOP-LEVEL parameter, NOT in messages array:
  client.messages.create(model=model, max_tokens=4096, system=system_prompt, messages=messages, tools=[tooldef_to_anthropic(t) for t in tools], temperature=0.7)
- If system_prompt is None, omit the system parameter.
- Response parsing:
  - response.content is a LIST of content blocks
  - Text: blocks where type == "text" → concatenate .text fields
  - Tool use: blocks where type == "tool_use" → ToolCall(name=block.name, arguments=block.input, id=block.id)
  - response.stop_reason: "end_turn" means text response; "tool_use" means tool calls
  - response.usage.input_tokens → prompt_tokens
  - response.usage.output_tokens → completion_tokens
  - prefill_duration_ms = None (always, for Anthropic)

Step 2 — Update append_tool_interaction in llm_client.py for Anthropic:

The llm_client helper handles provider-specific message formatting. The agent loop needs ONE small update: pass the system prompt to chat().

Update agent_loop.py:
- Before the loop: system_for_api = llm_client.get_system_prompt_for_api(system_prompt)
- In the chat call: response = llm_client.chat(messages, tools, model, temperature, system=system_for_api)
- Everything else in agent_loop stays the same — append_tool_interaction already encapsulates the format differences.

For append_tool_interaction, add the Anthropic branch:
1. Append ONE assistant message with the full content blocks: {"role": "assistant", "content": [<all content blocks from response>]}
2. Collect ALL tool results
3. Append ONE user message with ALL tool_result blocks: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": result} for each tool call]}

CRITICAL: Do NOT append separate user messages per tool result. Anthropic requires strictly alternating user/assistant messages. Two user messages in a row → 400 error.

Also update build_initial_messages for Anthropic:
- For Anthropic: return only [{"role": "user", "content": user_prompt}] (system goes as top-level param, not in messages)

Also update get_system_prompt_for_api:
- For Anthropic: return the system_prompt string (used as top-level param)
- For Ollama: return None (system is already in messages array)

Step 3 — Update placements.py:

Add a function get_placements(provider) that returns:
- For both "ollama" and "anthropic": [system, user, tool_description] — all three

Placements do NOT need to change for Anthropic. They produce PlacementConfig with list[ToolDef] — the llm_client handles conversion to provider-specific format. This is already the architecture from Prompt 1.

Step 4 — Update runner.py:

- Read LLM_PROVIDER from config
- Get applicable placements from get_placements(provider)
- Set runs_per_placement: 50 for ollama, 20 for anthropic
- Everything else stays the same

Step 5 — Update config.py:

Add: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default "claude-sonnet-4-6").
If LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY is not set, print clear error and exit.

Step 6 — Update .env.example:

Add:
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6

Step 7 — Verify:

If you have an Anthropic API key available:
1. Set LLM_PROVIDER=anthropic in .env
2. Run --smoke-test (should do 3 runs: system, user, tool_description)
3. Confirm 3 new rows in CSV with provider="anthropic"
4. Run analyze — should show both providers in the table and chart

If no API key available: verify the code imports cleanly, config validation catches the missing key with a clear error, and analyze handles a CSV with only ollama rows gracefully.
```

**Verify:** Anthropic smoke test passes (or at minimum, the code structure is correct and fails only on missing API key).

---

## Summary

| Prompt | What it builds | Verify by |
|--------|---------------|-----------|
| 1 | Project + Ollama client + tools + agent loop | One successful agent run with correct TODO counts |