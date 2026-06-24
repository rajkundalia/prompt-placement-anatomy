"""Phase 2 analysis module — Hierarchy Resolution.

Reads results/phase2/runs.csv, computes winner distribution statistics per
provider, prints a summary table to stdout, and saves a horizontal bar chart
to results/phase2/chart.png.

Usage:
    python -m prompt_placement_anatomy.phase2.analyze_phase2
"""

import logging
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results") / "phase2"
CSV_PATH = RESULTS_DIR / "runs.csv"
CHART_PATH = RESULTS_DIR / "chart.png"

Z95 = 1.96  # z-score for 95% confidence interval

# Canonical display labels for each outcome, in chart order (top to bottom)
OUTCOME_ORDER = ["system", "user", "tool", "none", "conflict-in-output"]
OUTCOME_LABELS = {
    "system": "system [DONE]",
    "user": "user [FINISHED]",
    "tool": "tool [COMPLETE]",
    "none": "none",
    "conflict-in-output": "conflict-in-output",
}

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int) -> tuple[float, float, float]:
    """Compute Wilson 95% confidence interval for a proportion.

    Args:
        successes: Number of successes (runs with this outcome).
        total: Total number of trials.

    Returns:
        (proportion, lower_bound, upper_bound) — all NaN if total is 0.
    """
    if total == 0:
        return math.nan, math.nan, math.nan
    proportion = successes / total
    denominator = 1 + Z95**2 / total
    center = (proportion + Z95**2 / (2 * total)) / denominator
    margin = Z95 * math.sqrt(proportion * (1 - proportion) / total + Z95**2 / (4 * total**2)) / denominator
    return proportion, max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and preprocess the Phase 2 runs CSV.

    Returns:
        DataFrame with typed columns.

    Raises:
        SystemExit: If the CSV does not exist.
    """
    if not CSV_PATH.exists():
        logger.error("No results found at %s. Run the experiment first.", CSV_PATH)
        raise SystemExit(1)

    dataframe = pd.read_csv(CSV_PATH)

    for col in ["turns", "total_tokens", "prefill_ms_turn_1", "prefill_ms_subsequent_mean"]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

    return dataframe


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_provider_stats(group: pd.DataFrame) -> dict:
    """Compute all statistics for a single provider group.

    Reports winner distribution (frequency + Wilson CI) for each of the 5
    possible outcomes, computed over successful runs only.

    Handles zero-success groups gracefully — returns N/A strings for metrics
    that require at least one successful run.

    Args:
        group: DataFrame subset for one provider.

    Returns:
        Dict of computed stats.
    """
    total = len(group)
    n_success = int((group["status"] == "success").sum())

    base: dict = {
        "total_runs": total,
        "n_success": n_success,
    }

    if n_success == 0:
        winner_stats: dict = {}
        for outcome in OUTCOME_ORDER:
            winner_stats[f"count_{outcome}"] = 0
            winner_stats[f"freq_{outcome}"] = "N/A"
            winner_stats[f"ci_lo_{outcome}"] = "N/A"
            winner_stats[f"ci_hi_{outcome}"] = "N/A"
        success_stats: dict = {
            "mean_turns": "N/A",
            "mean_total_tokens": "N/A",
            "mean_prefill_turn_1": "N/A",
        }
        return {**base, **winner_stats, **success_stats}

    success_rows = group[group["status"] == "success"]

    winner_stats = {}
    for outcome in OUTCOME_ORDER:
        count = int((success_rows["winner"] == outcome).sum())
        freq, ci_lo, ci_hi = wilson_ci(count, n_success)
        winner_stats[f"count_{outcome}"] = count
        winner_stats[f"freq_{outcome}"] = freq
        winner_stats[f"ci_lo_{outcome}"] = ci_lo
        winner_stats[f"ci_hi_{outcome}"] = ci_hi

    def safe_mean(series: pd.Series) -> float | str:
        vals = series.dropna().tolist()
        return statistics.mean(vals) if vals else "N/A"

    success_stats = {
        "mean_turns": safe_mean(success_rows["turns"]),
        "mean_total_tokens": safe_mean(success_rows["total_tokens"]),
        "mean_prefill_turn_1": safe_mean(success_rows["prefill_ms_turn_1"]),
    }

    return {**base, **winner_stats, **success_stats}


def build_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate winner distribution stats for all provider groups.

    Args:
        dataframe: Full raw DataFrame from load_data().

    Returns:
        Summary DataFrame with one row per provider.
    """
    rows = []
    for provider, group in dataframe.groupby("provider"):
        stats = compute_provider_stats(group)
        rows.append({"provider": provider, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def save_chart(summary: pd.DataFrame) -> None:
    """Generate a horizontal grouped bar chart of winner frequencies and save to CHART_PATH.

    Each outcome gets one bar per provider. X-axis shows frequency (0–1)
    with Wilson CI error bars. Y-axis shows outcome labels.

    Args:
        summary: Summary DataFrame from build_summary().
    """
    providers = sorted(summary["provider"].unique().tolist())
    n_providers = len(providers)
    n_outcomes = len(OUTCOME_ORDER)

    fig, ax = plt.subplots(figsize=(9, max(4, n_outcomes * 1.2 * n_providers)))

    bar_height = 0.7 / max(n_providers, 1)
    colors = [f"C{i}" for i in range(n_providers)]

    for provider_idx, provider in enumerate(providers):
        provider_row = summary[summary["provider"] == provider]
        if provider_row.empty:
            continue

        freqs = []
        err_lo = []
        err_hi = []

        for outcome in OUTCOME_ORDER:
            freq_val = provider_row[f"freq_{outcome}"].values[0]
            ci_lo_val = provider_row[f"ci_lo_{outcome}"].values[0]
            ci_hi_val = provider_row[f"ci_hi_{outcome}"].values[0]

            # Handle N/A (zero-success) gracefully
            try:
                freq_f = float(freq_val)
                ci_lo_f = float(ci_lo_val)
                ci_hi_f = float(ci_hi_val)
            except (TypeError, ValueError):
                freq_f, ci_lo_f, ci_hi_f = 0.0, 0.0, 0.0

            if math.isnan(freq_f):
                freq_f, ci_lo_f, ci_hi_f = 0.0, 0.0, 0.0

            freqs.append(freq_f)
            err_lo.append(max(0.0, freq_f - ci_lo_f))
            err_hi.append(max(0.0, ci_hi_f - freq_f))

        # Y positions: one slot per outcome, offset per provider
        offset = (provider_idx - (n_providers - 1) / 2) * bar_height
        y_positions = [i + offset for i in range(n_outcomes)]

        ax.barh(
            y_positions,
            freqs,
            height=bar_height,
            label=provider,
            xerr=[err_lo, err_hi],
            capsize=4,
            color=colors[provider_idx],
        )

        # Annotate bars with percentage values
        for y_pos, freq_f, err_hi_val in zip(y_positions, freqs, err_hi):
            if freq_f > 0 or err_hi_val > 0:
                ax.text(
                    freq_f + err_hi_val + 0.01,
                    y_pos,
                    f"{freq_f:.0%}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_yticks(range(n_outcomes))
    ax.set_yticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER], fontsize=10)
    ax.set_xlabel("Frequency (proportion of successful runs)", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(
        "Phase 2 — Hierarchy Resolution: Winner Distribution",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # top outcome at the top

    if n_providers > 1:
        ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved to %s", CHART_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyze() -> None:
    """Load results, print summary table, and save chart."""
    dataframe = load_data()
    summary = build_summary(dataframe)

    # Select display columns — winner counts + frequencies
    display_cols = ["provider", "total_runs", "n_success"]
    for outcome in OUTCOME_ORDER:
        display_cols.append(f"count_{outcome}")
        display_cols.append(f"freq_{outcome}")
    display_cols += ["mean_turns", "mean_total_tokens", "mean_prefill_turn_1"]

    available_cols = [col for col in display_cols if col in summary.columns]
    print("\n" + summary[available_cols].to_string(index=False))
    print()

    save_chart(summary)
    logger.info("Analysis complete.")


def main() -> None:
    """Configure logging and run analysis."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    analyze()


if __name__ == "__main__":
    main()
