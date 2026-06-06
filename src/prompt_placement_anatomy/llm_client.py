"""LLM client abstraction for the placement experiment.

Provides a unified interface for Ollama (local) and Anthropic (cloud) providers.
Provider is selected via config.LLM_PROVIDER.

Public interface:
    chat()                    -- send messages, get a normalized LLMResponse
    build_initial_messages()  -- construct the initial messages list
    append_tool_interaction() -- append tool call + results to message history
    get_system_prompt_for_api() -- return system prompt for API (Anthropic only)
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx
import ollama
from anthropic import Anthropic

from prompt_placement_anatomy import config
from prompt_placement_anatomy.tools import ToolDef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call returned by the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class Usage:
    """Token usage for a single LLM call."""

    prompt_tokens: int  # input tokens
    completion_tokens: int  # output tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Attributes:
        content: Text content of the response. Empty string if only tool calls.
        tool_calls: Tool calls requested by the model (empty list if none).
        usage: Token usage for this call.
        prefill_duration_ms: Time-to-first-token in ms (Ollama only; None for Anthropic).
        raw_response: Provider-specific object for message history reconstruction.
            Ollama: the response.message object (appended directly to history).
            Anthropic: the response.content list of blocks (used in assistant message).
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    prefill_duration_ms: float | None = None
    raw_response: Any = None


# ---------------------------------------------------------------------------
# Tool Definition Format converters
# ---------------------------------------------------------------------------


def tooldef_to_ollama(tool: ToolDef) -> dict[str, Any]:
    """Convert a ToolDef to the Ollama wire format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def tooldef_to_anthropic(tool: ToolDef) -> dict[str, Any]:
    """Convert a ToolDef to the Anthropic wire format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


# ---------------------------------------------------------------------------
# Ollama provider (private)
# ---------------------------------------------------------------------------

_ollama_client: ollama.Client | None = None
_ollama_checked_models: set[str] = set()


def _get_ollama_client() -> ollama.Client:
    """Return (or lazily create) the shared Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_HOST)
    return _ollama_client


def _ensure_ollama_model(model: str) -> None:
    """Verify the model exists in Ollama; exit with code 1 if not.

    Result is cached per model name so the check only runs once per process.
    """
    if model in _ollama_checked_models:
        return
    client = _get_ollama_client()
    try:
        client.show(model)
        _ollama_checked_models.add(model)
    except httpx.ConnectError:
        logger.error(
            "Ollama is not running at %s. Start it with `ollama serve`.",
            config.OLLAMA_HOST,
        )
        sys.exit(1)
    except Exception:  # noqa: BLE001
        logger.error("Model '%s' not found. Run: ollama pull %s", model, model)
        sys.exit(1)


def _chat_ollama(
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    model: str,
    temperature: float,
) -> LLMResponse:
    """Send a chat request to Ollama and return a normalized LLMResponse."""
    _ensure_ollama_model(model)
    client = _get_ollama_client()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=[tooldef_to_ollama(tool) for tool in tools],
            options={"temperature": temperature},
        )
    except httpx.ConnectError:
        logger.error(
            "Ollama is not running at %s. Start it with `ollama serve`.",
            config.OLLAMA_HOST,
        )
        sys.exit(1)

    # content can be None when the response is tool-calls-only
    content: str = response.message.content or ""

    # tool_calls can be None (not an empty list) — must use truthy check
    tool_calls: list[ToolCall] = []
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            tool_calls.append(
                ToolCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    id=None,  # Ollama does not provide tool-call IDs
                )
            )

    prompt_tokens: int = response.prompt_eval_count or 0
    completion_tokens: int = response.eval_count or 0
    prefill_ns = response.prompt_eval_duration
    prefill_ms: float | None = prefill_ns / 1_000_000 if prefill_ns is not None else None

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        prefill_duration_ms=prefill_ms,
        raw_response=response.message,  # append directly to Ollama history
    )


# ---------------------------------------------------------------------------
# Anthropic provider (private)
# ---------------------------------------------------------------------------


def _chat_anthropic(
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    model: str,
    temperature: float,
    system: str | None,
) -> LLMResponse:
    """Send a chat request to Anthropic and return a normalized LLMResponse."""
    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set. Add it to .env or set the environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system or "",
        messages=messages,
        tools=[tooldef_to_anthropic(tool) for tool in tools],
        temperature=temperature,
    )

    content = ""
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=block.name,
                    arguments=block.input,
                    id=block.id,
                )
            )

    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        prefill_duration_ms=None,  # not available for Anthropic
        raw_response=response.content,  # list of content blocks for assistant message
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def chat(
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    model: str,
    temperature: float = 0.7,
    system: str | None = None,
) -> LLMResponse:
    """Send messages to the active LLM provider and return a normalized response.

    Provider is determined by config.LLM_PROVIDER. ToolDef objects are
    converted to the provider-specific wire format internally.

    Args:
        messages: Message history in the provider-appropriate format.
        tools: Provider-agnostic tool definitions.
        model: Model name (provider-specific).
        temperature: Sampling temperature (default 0.7).
        system: System prompt. For Anthropic, passed as top-level API param.
                For Ollama, ignored here (system is already in messages via
                build_initial_messages).

    Returns:
        Normalized LLMResponse.
    """
    if config.LLM_PROVIDER == "ollama":
        return _chat_ollama(messages, tools, model, temperature)
    return _chat_anthropic(messages, tools, model, temperature, system)


def build_initial_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    """Construct the initial messages list from system and user prompts.

    For Ollama: includes a system message followed by the user message.
    For Anthropic: includes only the user message (system is a top-level API param).

    Args:
        system_prompt: The system instruction string.
        user_prompt: The initial user task string.

    Returns:
        List of message dicts ready to pass to chat().
    """
    if config.LLM_PROVIDER == "ollama":
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    # Anthropic: system goes as top-level param in chat(); not in messages array
    return [{"role": "user", "content": user_prompt}]


def append_tool_interaction(
    messages: list[dict[str, Any]],
    response: LLMResponse,
    tool_calls: list[ToolCall],
    results: list[str],
) -> None:
    """Append the assistant's tool call and tool results to the message history in-place.

    Handles provider-specific message format differences:
    - Ollama: appends the raw assistant message, then one {"role": "tool"} per result.
    - Anthropic: appends ONE assistant message with all content blocks, then ONE user
      message containing ALL tool_result blocks. This satisfies Anthropic's requirement
      for strictly alternating user/assistant messages.

    Args:
        messages: The message history list to mutate.
        response: The LLMResponse that contained tool calls.
        tool_calls: The tool calls that were dispatched.
        results: The corresponding string results from dispatch_tool().
    """
    if config.LLM_PROVIDER == "ollama":
        messages.append(response.raw_response)  # original Ollama message object
        for result in results:
            messages.append({"role": "tool", "content": result})
    else:
        # Anthropic: one assistant message with all content blocks
        messages.append({"role": "assistant", "content": response.raw_response})
        # Then one user message with ALL tool_result blocks — never split across messages
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                    for tc, result in zip(tool_calls, results)
                ],
            }
        )


def get_system_prompt_for_api(system_prompt: str) -> str | None:
    """Return the system prompt for the provider's top-level API parameter.

    For Anthropic: returns the system_prompt string (passed to messages.create).
    For Ollama: returns None (system prompt is already in the messages array).

    Args:
        system_prompt: The system instruction string.

    Returns:
        The system prompt string for Anthropic, or None for Ollama.
    """
    if config.LLM_PROVIDER == "anthropic":
        return system_prompt
    return None
