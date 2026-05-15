"""Tool definitions and dispatcher for the placement experiment.

Defines the provider-agnostic ToolDef dataclass, the two filesystem tools
(list_files, read_file), and the dispatch_tool function used by the agent loop.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    """Provider-agnostic tool definition.

    Used by placements.py to build PlacementConfig and by llm_client.py to
    convert to provider-specific wire formats (Ollama / Anthropic).
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def list_files(directory: str) -> list[str]:
    """Return a sorted list of file paths in the given directory.

    Args:
        directory: Path to the directory to list.

    Returns:
        Sorted list of absolute file path strings.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        logger.warning("list_files: %s is not a directory", directory)
        return []
    return sorted(str(path) for path in dir_path.iterdir() if path.is_file())


def read_file(path: str) -> str:
    """Return the full contents of the file at the given path.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as a UTF-8 string, or an error message if not found.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        logger.warning("read_file: %s does not exist or is not a file", path)
        return f"Error: file not found: {path}"
    return file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH_TABLE: dict[str, Callable[[dict], object]] = {
    "list_files": lambda args: list_files(args["directory"]),
    "read_file": lambda args: read_file(args["path"]),
}


def dispatch_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name and return the result as a string.

    If the tool returns a list (list_files), it is joined with newlines.
    If the tool name is unknown, an error string is returned — no crash.

    Args:
        name: Tool name registered in the dispatch table.
        arguments: Tool arguments dict (from the LLM tool call).

    Returns:
        Tool result as a string.
    """
    if name not in _DISPATCH_TABLE:
        logger.warning("dispatch_tool: unknown tool %r", name)
        return f"Error: unknown tool {name!r}"
    try:
        result = _DISPATCH_TABLE[name](arguments)
        if isinstance(result, list):
            return "\n".join(result)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        logger.error("dispatch_tool: error executing %r: %s", name, exc)
        return f"Error executing tool {name!r}: {exc}"


# ---------------------------------------------------------------------------
# Tool registry (provider-agnostic)
# ---------------------------------------------------------------------------

TOOLS: list[ToolDef] = [
    ToolDef(
        name="list_files",
        description="Returns a sorted list of file paths in the given directory.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory",
                }
            },
            "required": ["directory"],
        },
    ),
    ToolDef(
        name="read_file",
        description="Returns the full contents of the file at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file",
                }
            },
            "required": ["path"],
        },
    ),
]
