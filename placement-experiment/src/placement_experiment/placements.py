"""Placement variant definitions for the experiment.

Each variant places the [DONE] instruction in exactly one prompt slot:
- system:           appended to the system message
- user:             prepended to the user message
- tool_description: appended to the read_file tool description (deep-copied)
"""

import copy
import logging
from dataclasses import dataclass

from placement_experiment import config
from placement_experiment.tools import TOOLS, ToolDef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SYSTEM = "You are a helpful assistant that uses tools to analyze files."
BASE_USER = "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."
INSTRUCTION = "End your final answer with the marker [DONE]"

# ---------------------------------------------------------------------------
# PlacementConfig
# ---------------------------------------------------------------------------


@dataclass
class PlacementConfig:
    """Provider-agnostic configuration for a single experiment variant.

    Attributes:
        system_prompt: The system instruction string.
        user_prompt:   The initial user task string.
        tools:         Provider-agnostic tool definitions (llm_client converts to wire format).
        model:         Model name for the active provider.
        name:          Variant identifier — "system", "user", or "tool_description".
    """

    system_prompt: str
    user_prompt: str
    tools: list[ToolDef]
    model: str
    name: str


# ---------------------------------------------------------------------------
# Variant factories
# ---------------------------------------------------------------------------


def system_placement() -> PlacementConfig:
    """Return the 'system' variant — instruction appended to the system message."""
    return PlacementConfig(
        system_prompt=f"{BASE_SYSTEM} {INSTRUCTION}",
        user_prompt=BASE_USER,
        tools=TOOLS,
        model=config.active_model(),
        name="system",
    )


def user_placement() -> PlacementConfig:
    """Return the 'user' variant — instruction prepended to the user message."""
    return PlacementConfig(
        system_prompt=BASE_SYSTEM,
        user_prompt=f"{INSTRUCTION}\n\n{BASE_USER}",
        tools=TOOLS,
        model=config.active_model(),
        name="user",
    )


def tool_description_placement() -> PlacementConfig:
    """Return the 'tool_description' variant — instruction appended to read_file's description.

    Uses copy.deepcopy() to avoid mutating the shared TOOLS list. The instruction
    is injected only into the read_file tool, which is the last tool the agent
    calls before producing its final answer.
    """
    tools_copy = copy.deepcopy(TOOLS)
    for tool in tools_copy:
        if tool.name == "read_file":
            tool.description = f"{tool.description} {INSTRUCTION}"
            break
    else:
        logger.warning("tool_description_placement: 'read_file' tool not found in TOOLS")

    return PlacementConfig(
        system_prompt=BASE_SYSTEM,
        user_prompt=BASE_USER,
        tools=tools_copy,
        model=config.active_model(),
        name="tool_description",
    )


def get_all_placements() -> list[PlacementConfig]:
    """Return all three placement variants in a fixed order."""
    return [
        system_placement(),
        user_placement(),
        tool_description_placement(),
    ]
