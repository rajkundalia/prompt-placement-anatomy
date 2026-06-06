"""Minimal agent loop for the placement experiment.

Runs a while-loop calling the LLM until it returns a text response with no
tool calls (the final answer), or the turn cap (MAX_TURNS) is reached.

The loop is fully provider-agnostic: all message construction and history
appending goes through llm_client helper functions. No provider-specific
message dicts are built here.
"""

import logging
from dataclasses import dataclass, field

from prompt_placement_anatomy import llm_client
from prompt_placement_anatomy.placements import PlacementConfig
from prompt_placement_anatomy.tools import dispatch_tool

logger = logging.getLogger(__name__)

MAX_TURNS = 15

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """Token usage and timing captured for a single LLM call."""

    turn: int
    prompt_tokens: int
    completion_tokens: int
    prefill_duration_ms: float | None  # Ollama only; None for Anthropic


@dataclass
class AgentResult:
    """Result of a complete agent loop run.

    Attributes:
        status:           "success" | "timeout" | "error"
        turns:            Total number of LLM calls made.
        final_answer:     The agent's last text response (None on timeout).
        per_turn_records: Per-call token and timing data.
    """

    status: str
    turns: int
    final_answer: str | None
    per_turn_records: list[TurnRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run(placement: PlacementConfig) -> AgentResult:
    """Run the agent loop for a given placement configuration.

    Continues until the model returns a text-only response (success) or
    the turn cap is hit (timeout). Any exception from the LLM client is
    caught and returned as an error result so the runner can log and continue.

    Args:
        placement: PlacementConfig with system/user prompts, tools, and model.

    Returns:
        AgentResult with status, turn count, final answer, and per-turn records.
    """
    turn_count = 0
    per_turn_records: list[TurnRecord] = []

    messages = llm_client.build_initial_messages(placement.system_prompt, placement.user_prompt)
    system_for_api = llm_client.get_system_prompt_for_api(placement.system_prompt)

    status = "timeout"
    final_answer: str | None = None

    while turn_count < MAX_TURNS:
        # -- LLM call --------------------------------------------------------
        try:
            response = llm_client.chat(
                messages=messages,
                tools=placement.tools,
                model=placement.model,
                temperature=0.7,
                system=system_for_api,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed on turn %d: %s", turn_count, exc)
            return AgentResult(
                status="error",
                turns=turn_count,
                final_answer=str(exc),
                per_turn_records=per_turn_records,
            )

        # -- Record per-turn metrics -----------------------------------------
        per_turn_records.append(
            TurnRecord(
                turn=turn_count,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                prefill_duration_ms=response.prefill_duration_ms,
            )
        )
        turn_count += 1

        # -- Route response --------------------------------------------------
        if response.tool_calls:
            # Dispatch every tool call; collect results as strings.
            # If response also has text content alongside tool_calls, ignore it —
            # tool_calls take priority (per spec).
            results = [dispatch_tool(toolcall.name, toolcall.arguments) for toolcall in response.tool_calls]
            llm_client.append_tool_interaction(messages, response, response.tool_calls, results)
            continue

        # No tool calls — this is the final text response.
        final_answer = response.content
        status = "success"
        break

    if status == "timeout":
        logger.warning("Agent reached turn cap (%d) without a final answer.", MAX_TURNS)

    return AgentResult(
        status=status,
        turns=turn_count,
        final_answer=final_answer,
        per_turn_records=per_turn_records,
    )
