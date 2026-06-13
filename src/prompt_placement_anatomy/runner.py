"""Experiment runner for the placement experiment.

Executes agent trials across all placement variants, writes results to a
resumable CSV, and logs progress. Supports a --smoke-test flag for quick
pipeline validation (1 run per placement).

All scripts assume CWD is the project root.

Usage:
    python -m prompt_placement_anatomy.runner
    python -m prompt_placement_anatomy.runner --smoke-test
"""

import argparse
import csv
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import ollama

from prompt_placement_anatomy import config
from prompt_placement_anatomy.agent_loop import AgentResult
from prompt_placement_anatomy.agent_loop import run as run_agent
from prompt_placement_anatomy.placements import PlacementConfig, get_all_placements

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "runs.csv"

OLLAMA_RUNS_PER_PLACEMENT = 50
ANTHROPIC_RUNS_PER_PLACEMENT = 50

CSV_COLUMNS = [
    "provider",
    "placement",
    "run_id",
    "status",
    "turns",
    "total_tokens",
    "prefill_ms_turn_1",
    "prefill_ms_subsequent_mean",
    "compliance",
    "final_answer",
]

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_completed_runs() -> set[tuple[str, str, int]]:
    """Read the existing CSV and return a set of already-completed (provider, placement, run_id).

    Returns:
        Set of tuples identifying runs that should be skipped.
    """
    if not CSV_PATH.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                completed.add((row["provider"], row["placement"], int(row["run_id"])))
            except (KeyError, ValueError):
                pass  # skip malformed rows
    logger.info("Resuming: found %d completed runs in %s", len(completed), CSV_PATH)
    return completed


def open_csv_writer() -> tuple[Any, Any]:
    """Open (or create) the CSV file for appending.

    Returns:
        (writer, file_handle) — caller must close the file handle.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    file_handle = CSV_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()
    return writer, file_handle


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_compliance(result: AgentResult) -> bool | None:
    """Return True/False for successful runs, None for non-success.

    Args:
        result: AgentResult from the agent loop.

    Returns:
        True if [DONE] found in last 80 chars of final answer, False if not,
        None if the run did not succeed.
    """
    if result.status != "success" or result.final_answer is None:
        return None
    return bool(re.search(r"\[done\]", result.final_answer[-80:], re.IGNORECASE))


def compute_prefill_metrics(result: AgentResult) -> tuple[float | None, float | None]:
    """Return (prefill_ms_turn_1, prefill_ms_subsequent_mean) from per-turn records.

    Args:
        result: AgentResult with per_turn_records populated.

    Returns:
        Tuple of (turn-1 prefill ms, mean subsequent prefill ms).
        Both are None for Anthropic (prefill_duration_ms is always None there).
    """
    records = result.per_turn_records
    turn_1 = records[0].prefill_duration_ms if records else None

    subsequent_ms = [r.prefill_duration_ms for r in records[1:] if r.prefill_duration_ms is not None]
    subsequent_mean = statistics.mean(subsequent_ms) if subsequent_ms else None

    return turn_1, subsequent_mean


def build_csv_row(placement: PlacementConfig, run_id: int, result: AgentResult) -> dict[str, Any]:
    """Build a CSV row dict from a completed agent run.

    Args:
        placement: The PlacementConfig used for this run.
        run_id:    The run index.
        result:    The AgentResult from agent_loop.run().

    Returns:
        Dict matching CSV_COLUMNS.
    """
    compliance = compute_compliance(result)
    turn_1_ms, subsequent_ms = compute_prefill_metrics(result)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in result.per_turn_records)

    final_answer_str = (result.final_answer or "").replace("\n", " ")[:500]

    return {
        "provider": config.LLM_PROVIDER,
        "placement": placement.name,
        "run_id": run_id,
        "status": result.status,
        "turns": result.turns,
        "total_tokens": total_tokens,
        "prefill_ms_turn_1": "" if turn_1_ms is None else turn_1_ms,
        "prefill_ms_subsequent_mean": "" if subsequent_ms is None else subsequent_ms,
        "compliance": "" if compliance is None else compliance,
        "final_answer": final_answer_str,
    }


# ---------------------------------------------------------------------------
# Pre-flight: model check + warm-up
# ---------------------------------------------------------------------------


def preflight_checks() -> None:
    """Verify required models/credentials are available. Warm up Ollama if applicable.

    Exits with code 1 if a required resource is missing.
    """
    if config.LLM_PROVIDER == "ollama":
        client = ollama.Client(host=config.OLLAMA_HOST)
        try:
            client.show(config.OLLAMA_MODEL)
        except httpx.ConnectError:
            logger.error(
                "Ollama is not running at %s. Start it with `ollama serve`.",
                config.OLLAMA_HOST,
            )
            sys.exit(1)
        except Exception:  # noqa: BLE001
            logger.error("Model '%s' not found. Run: ollama pull %s", config.OLLAMA_MODEL, config.OLLAMA_MODEL)
            sys.exit(1)

        # Warm-up: force model into memory so turn-1 prefill isn't inflated by a cold start.
        logger.info("Warming up model '%s' (forcing into memory)...", config.OLLAMA_MODEL)
        try:
            client.chat(model=config.OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}])
            logger.info("Warm-up complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up request failed (non-fatal): %s", exc)

    else:  # anthropic
        if not config.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_API_KEY not set. Add it to .env or set the environment variable.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Tracks per-placement compliance rates and overall ETA."""

    def __init__(self, placements: list[PlacementConfig], total_runs: int) -> None:
        self.total_runs = total_runs
        self.completed = 0
        self.start_time = time.monotonic()
        self.stats: dict[str, dict[str, int]] = {p.name: {"runs": 0, "compliant": 0} for p in placements}

    def record(self, placement_name: str, compliance: bool | None) -> None:
        """Record one completed run."""
        self.completed += 1
        self.stats[placement_name]["runs"] += 1
        if compliance is True:
            self.stats[placement_name]["compliant"] += 1

    def maybe_log(self) -> None:
        """Log progress if we've hit a multiple of 10 completed runs."""
        if self.completed % 10 != 0:
            return
        elapsed = time.monotonic() - self.start_time
        logger.info("--- Progress: %d / %d runs ---", self.completed, self.total_runs)
        for name, placement_stats in self.stats.items():
            if placement_stats["runs"] > 0:
                rate = placement_stats["compliant"] / placement_stats["runs"] * 100
                logger.info("  %-20s  %d runs  %.0f%% compliant", name, placement_stats["runs"], rate)
        if self.completed >= 10:
            eta_s = elapsed / self.completed * (self.total_runs - self.completed)
            logger.info("  ETA: %.0f min remaining", eta_s / 60)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(smoke_test: bool = False) -> None:
    """Run all placement trials and write results to CSV.

    Args:
        smoke_test: If True, run only 1 trial per placement (3 total).
    """
    runs_per_placement = 1 if smoke_test else (
        OLLAMA_RUNS_PER_PLACEMENT if config.LLM_PROVIDER == "ollama" else ANTHROPIC_RUNS_PER_PLACEMENT
    )

    preflight_checks()

    placements = get_all_placements()
    completed_runs = load_completed_runs()

    # Build the list of (placement, run_id) pairs that still need to be executed.
    # Skip any pair already recorded in the CSV (resumable behaviour).
    pending: list[tuple[PlacementConfig, int]] = []
    for placement in placements:
        for run_id in range(runs_per_placement):
            already_done = (config.LLM_PROVIDER, placement.name, run_id) in completed_runs
            if not already_done:
                pending.append((placement, run_id))

    if not pending:
        logger.info("All runs already completed. Nothing to do.")
        return

    logger.info(
        "Starting %d runs (%s provider, %d per placement, %d already done).",
        len(pending),
        config.LLM_PROVIDER,
        runs_per_placement,
        len(completed_runs),
    )

    tracker = ProgressTracker(placements, len(pending))
    writer, csv_file = open_csv_writer()
    smoke_compliance_results: list[bool | None] = []

    try:
        for placement, run_id in pending:
            logger.info("Running: placement=%s  run_id=%d", placement.name, run_id)

            try:
                result = run_agent(placement)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error in run %s/%d: %s", placement.name, run_id, exc)
                from prompt_placement_anatomy.agent_loop import AgentResult  # avoid circular at module level

                result = AgentResult(status="error", turns=0, final_answer=str(exc))

            compliance = compute_compliance(result)
            row = build_csv_row(placement, run_id, result)
            writer.writerow(row)
            csv_file.flush()  # persist immediately so partial progress survives crashes

            tracker.record(placement.name, compliance)
            tracker.maybe_log()

            if smoke_test:
                smoke_compliance_results.append(compliance)

    finally:
        csv_file.close()

    logger.info("Done. %d runs written to %s.", len(pending), CSV_PATH)

    # Smoke-test post-check: warn if zero compliant runs
    if smoke_test:
        successful_compliance = [c for c in smoke_compliance_results if c is not None]
        if successful_compliance and not any(successful_compliance):
            logger.warning(
                "Warning: No runs were compliant. The model may not follow format instructions "
                "reliably. Consider trying a larger model or a different instruction before "
                "running the full experiment."
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and start the experiment."""
    parser = argparse.ArgumentParser(description="Run the prompt placement experiment.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 trial per placement (3 total) to verify the pipeline end-to-end.",
    )
    args = parser.parse_args()
    run_experiment(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
