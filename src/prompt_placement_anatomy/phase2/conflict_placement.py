"""Phase 2 conflict placement definition and winner detection.

In Phase 2, the system prompt, user message, and tool description each
contain a *different* format instruction at the same time — creating a
3-way conflict. The model must pick one. This module defines:
  - The single conflict PlacementConfig (get_conflict_placement)
  - The winner detection function (detect_winner)

The experiment observes which slot's marker appears in the last 150
characters of the model's final answer.
"""

import copy
import re
from typing import Literal

from prompt_placement_anatomy.placements import PlacementConfig
from prompt_placement_anatomy.tools import TOOLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SYSTEM = "You are a helpful assistant that uses tools to analyze files."
BASE_USER = "Find the number of TODO markers in each markdown file in `data/sample_files/`. Report the count per file."

SYSTEM_INSTRUCTION = "End your final answer with the marker [DONE]"
USER_INSTRUCTION = "End your final answer with the marker [FINISHED]"
TOOL_INSTRUCTION = "End your final answer with the marker [COMPLETE]"

# ---------------------------------------------------------------------------
# Winner detection
# ---------------------------------------------------------------------------

WinnerType = Literal["system", "user", "tool", "none", "conflict-in-output"]

MARKER_PATTERNS: dict[str, str] = {
    "system": r"\[done\]",
    "user":   r"\[finished\]",
    "tool":   r"\[complete\]",
}


def detect_winner(final_answer: str | None) -> WinnerType:
    """Detect which slot's marker appears in the tail of the final answer.

    Checks the last 150 characters of the final answer. Returns the winning
    slot name, 'none' if no marker is found, or 'conflict-in-output' if
    multiple markers are found.

    150 characters is used (wider than Phase 1's 80) because [FINISHED] and
    [COMPLETE] are longer tokens than [DONE].

    Args:
        final_answer: The agent's final text response, or None.

    Returns:
        One of: "system", "user", "tool", "none", "conflict-in-output".
    """
    if not final_answer:
        return "none"
    tail = final_answer[-150:]
    found = []
    for slot, pattern in MARKER_PATTERNS.items():
        if re.search(pattern, tail, re.IGNORECASE):
            found.append(slot)
    if len(found) == 0:
        return "none"
    if len(found) == 1:
        return found[0]
    return "conflict-in-output"


# ---------------------------------------------------------------------------
# Conflict placement factory
# ---------------------------------------------------------------------------


def get_conflict_placement(model: str) -> PlacementConfig:
    """Return the single Phase 2 conflict placement config for the given model.

    All three slots carry conflicting format instructions simultaneously:
    - System: append [DONE]
    - User:   append [FINISHED]
    - Tool description (read_file): append [COMPLETE]

    Uses copy.deepcopy() to avoid mutating the shared TOOLS list.

    Args:
        model: Model name string for the active provider.

    Returns:
        PlacementConfig with all three conflicting instructions active.
    """
    tools = copy.deepcopy(TOOLS)
    for tool in tools:
        if tool.name == "read_file":
            tool.description = f"{tool.description} {TOOL_INSTRUCTION}"
    return PlacementConfig(
        system_prompt=f"{BASE_SYSTEM} {SYSTEM_INSTRUCTION}",
        user_prompt=f"{USER_INSTRUCTION}\n\n{BASE_USER}",
        tools=tools,
        model=model,
        name="conflict",
    )
