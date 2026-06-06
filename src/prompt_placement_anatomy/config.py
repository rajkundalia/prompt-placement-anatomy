"""Configuration module.

Loads settings from the .env file (or environment variables) and exposes them
as typed constants for the rest of the package.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file:
# src/prompt_placement_anatomy/config.py -> src/prompt_placement_anatomy/ -> src/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()

if LLM_PROVIDER not in {"ollama", "anthropic"}:
    print(f"ERROR: LLM_PROVIDER must be 'ollama' or 'anthropic', got: {LLM_PROVIDER!r}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# ---------------------------------------------------------------------------
# Anthropic settings
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def active_model() -> str:
    """Return the model name for the currently active provider."""
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    return ANTHROPIC_MODEL
