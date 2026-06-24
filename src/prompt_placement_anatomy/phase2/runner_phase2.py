"""Phase 2 experiment runner — Hierarchy Resolution.

Executes 30 agent trials under the 3-way conflict condition, writes results
to a resumable CSV, and logs progress including a running winner distribution.
Supports a --smoke-test flag for quick pipeline validation (1 run).

All commands must be run from the project root directory.

Usage:
    python -m prompt_placement_anatomy.phase2.runner_phase2
    python -m prompt_placement_anatomy.phase2.runner_phase2 --smoke-test
"""

import argparse
import csv
import logging
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
from prompt_placement_anatomy.phase2.conflict_placement import (
    WinnerType,
    detect_winner,
    get_conflict_placement,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results") / "phase2"
CSV_PATH = RESULTS_DIR / "runs.csv"

RUNS = 30

CSV_COLUMNS = [
    "provider",
    "run_id",
    "status",
    "turns",
    "total_tokens",
    "prefill_ms_turn_1",
    "prefill_ms_subsequent_mean",
    "winner",
    "final_answer",
]

VALID_WINNERS: set[WinnerType] = {"system", "user", "tool", "none", "conflict-in-output"}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_completed_runs() -> set[tuple[str, int]]:
    """Read the existing CSV and return a set of already-completed (provider, run_id) pairs.

    Returns:
        Set of tuples identifying runs that should be skipped.
    """
    if not CSV_PATH.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                completed.add((row["provider"], int(row["run_id"])))
            except (KeyError, ValueError):
                pass  # skip malformed rows
    logger.info("Resuming: found %d completed runs in %s", len(completed), CSV_PATH)
    return completed


def open_csv_writer() -> tuple[Any, Any]:
    """Open (or create) the Phase 2 CSV file for appending.

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

    subsequent_ms = [record.prefill_duration_ms for record in records[1:] if record.prefill_duration_ms is not None]
    subsequent_mean = statistics.mean(subsequent_ms) if subsequent_ms else None

    return turn_1, subsequent_mean


def build_csv_row(run_id: int, result: AgentResult, winner: WinnerType | str) -> dict[str, Any]:
    """Build a CSV row dict from a completed Phase 2 agent run.

    Args:
        run_id: The run index (0-based).
        result: The AgentResult from agent_loop.run().
        winner: The detected winner slot name, or empty string for non-success runs.

    Returns:
        Dict matching CSV_COLUMNS.
    """
    turn_1_ms, subsequent_ms = compute_prefill_metrics(result)
    total_tokens = sum(record.prompt_tokens + record.completion_tokens for record in result.per_turn_records)
    final_answer_str = (result.final_answer or "").replace("\n", " ")[:500]

    return {
        "provider": config.LLM_PROVIDER,
        "run_id": run_id,
        "status": result.status,
        "turns": result.turns,
        "total_tokens": total_tokens,
        "prefill_ms_turn_1": "" if turn_1_ms is None else turn_1_ms,
        "prefill_ms_subsequent_mean": "" if subsequent_ms is None else subsequent_ms,
        "winner": winner,
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


def log_progress(
    completed: int,
    total: int,
    winner_counts: dict[str, int],
    start_time: float,
) -> None:
    """Log progress at every 10-run boundary, including winner distribution.

    Args:
        completed:     Number of runs completed so far.
        total:         Total number of runs to complete.
        winner_counts: Running tally of winner outcomes.
        start_time:    Monotonic start time of the experiment.
    """
    if completed % 10 != 0:
        return
    elapsed = time.monotonic() - start_time
    logger.info("--- Progress: %d / %d runs ---", completed, total)

    total_outcomes = sum(winner_counts.values())
    if total_outcomes > 0:
        for outcome, count in sorted(winner_counts.items()):
            logger.info("  %-25s  %d  (%.0f%%)", outcome, count, count / total_outcomes * 100)

    if completed >= 10:
        eta_s = elapsed / completed * (total - completed)
        logger.info("  ETA: %.0f min remaining", eta_s / 60)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(smoke_test: bool = False) -> None:
    """Run Phase 2 conflict trials and write results to CSV.

    Args:
        smoke_test: If True, run only 1 trial.
    """
    total_runs = 1 if smoke_test else RUNS

    preflight_checks()

    completed_runs = load_completed_runs()
    model = config.active_model()
    placement = get_conflict_placement(model)

    pending = [
        run_id for run_id in range(total_runs)
        if (config.LLM_PROVIDER, run_id) not in completed_runs
    ]

    if not pending:
        logger.info("All runs already completed. Nothing to do.")
        return

    logger.info(
        "Starting %d runs (%s provider, %d already done).",
        len(pending),
        config.LLM_PROVIDER,
        len(completed_runs),
    )

    winner_counts: dict[str, int] = {outcome: 0 for outcome in VALID_WINNERS}
    start_time = time.monotonic()
    smoke_winner: WinnerType | None = None

    writer, csv_file = open_csv_writer()

    try:
        for loop_index, run_id in enumerate(pending):
            logger.info("Running: run_id=%d", run_id)

            try:
                result = run_agent(placement)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error in run %d: %s", run_id, exc)
                result = AgentResult(status="error", turns=0, final_answer=str(exc))

            if result.status == "success":
                winner: WinnerType | str = detect_winner(result.final_answer)
            else:
                winner = ""

            row = build_csv_row(run_id, result, winner)
            writer.writerow(row)
            csv_file.flush()  # persist immediately so partial progress survives crashes

            if winner in winner_counts:
                winner_counts[str(winner)] += 1

            completed_count = loop_index + 1
            log_progress(completed_count, len(pending), winner_counts, start_time)

            if smoke_test and result.status == "success":
                smoke_winner = winner if winner in VALID_WINNERS else "none"  # type: ignore[assignment]

    finally:
        csv_file.close()

    logger.info("Done. %d runs written to %s.", len(pending), CSV_PATH)

    # Smoke-test post-checks
    if smoke_test:
        logger.info("Smoke test winner: %s", smoke_winner)
        if smoke_winner == "none":
            logger.warning(
                "Warning: No marker detected. The model may not be following format instructions. "
                "Consider running the full experiment to confirm."
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and start the Phase 2 experiment."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 hierarchy resolution experiment (30 runs)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 trial to verify the pipeline end-to-end.",
    )
    args = parser.parse_args()
    run_experiment(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
