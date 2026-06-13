"""Analysis module for the placement experiment.

Reads results/runs.csv, computes statistics per (provider, placement) group,
prints a summary table to stdout, and saves a bar chart to results/chart.png.

Usage:
    python -m prompt_placement_anatomy.analyze
"""

import logging
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "runs.csv"
CHART_PATH = RESULTS_DIR / "chart.png"

Z95 = 1.96  # z-score for 95% confidence interval

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int) -> tuple[float, float, float]:
    """Compute Wilson 95% confidence interval for a proportion.

    Args:
        successes: Number of successes.
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


def std_error(values: list[float]) -> float:
    """Compute standard error of the mean.

    Args:
        values: List of numeric values.

    Returns:
        Standard error, or NaN if fewer than 2 values.
    """
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and preprocess the runs CSV.

    Returns:
        DataFrame with typed columns.
    """
    if not CSV_PATH.exists():
        logger.error("No results found at %s. Run the experiment first.", CSV_PATH)
        raise SystemExit(1)

    dataframe = pd.read_csv(CSV_PATH)

    # Normalise compliance column: "True" → True, "False" → False, empty → None
    def parse_compliance(value: object) -> bool | None:
        as_str = str(value).strip().lower()
        if as_str == "true":
            return True
        if as_str == "false":
            return False
        return None

    dataframe["compliance"] = dataframe["compliance"].map(parse_compliance)

    for col in ["turns", "total_tokens", "prefill_ms_turn_1", "prefill_ms_subsequent_mean"]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

    return dataframe


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_group_stats(group: pd.DataFrame) -> dict:
    """Compute all statistics for a single (provider, placement) group.

    Handles zero-completion groups gracefully — returns NaN for metrics that
    require at least one successful run.

    Args:
        group: DataFrame subset for one (provider, placement).

    Returns:
        Dict of computed stats.
    """
    total = len(group)
    n_success = int((group["status"] == "success").sum())
    completion_rate, cr_lo, cr_hi = wilson_ci(n_success, total)

    success_rows = group[group["status"] == "success"]

    nan_stats: dict = {
        "compliance_rate": math.nan, "compliance_rate_lo": math.nan, "compliance_rate_hi": math.nan,
        "mean_turns": math.nan, "se_turns": math.nan,
        "mean_total_tokens": math.nan, "se_total_tokens": math.nan,
        "mean_prefill_turn_1": math.nan, "se_prefill_turn_1": math.nan,
        "mean_prefill_subsequent": math.nan, "se_prefill_subsequent": math.nan,
    }

    base: dict = {
        "total_runs": total,
        "completions": n_success,
        "completion_rate": completion_rate,
        "completion_rate_lo": cr_lo,
        "completion_rate_hi": cr_hi,
    }

    if n_success == 0:
        return {**base, **nan_stats}

    # Compliance rate (among successful completions only)
    n_compliant = int(success_rows["compliance"].eq(True).sum())
    cmp_rate, cmp_lo, cmp_hi = wilson_ci(n_compliant, n_success)

    def mean_and_se(series: pd.Series) -> tuple[float, float]:
        vals = series.dropna().tolist()
        return (statistics.mean(vals) if vals else math.nan, std_error(vals))

    mean_turns, se_turns = mean_and_se(success_rows["turns"])
    mean_tokens, se_tokens = mean_and_se(success_rows["total_tokens"])
    mean_p1, se_p1 = mean_and_se(success_rows["prefill_ms_turn_1"])
    mean_psub, se_psub = mean_and_se(success_rows["prefill_ms_subsequent_mean"])

    return {
        **base,
        "compliance_rate": cmp_rate, "compliance_rate_lo": cmp_lo, "compliance_rate_hi": cmp_hi,
        "mean_turns": mean_turns, "se_turns": se_turns,
        "mean_total_tokens": mean_tokens, "se_total_tokens": se_tokens,
        "mean_prefill_turn_1": mean_p1, "se_prefill_turn_1": se_p1,
        "mean_prefill_subsequent": mean_psub, "se_prefill_subsequent": se_psub,
    }


def build_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats for all (provider, placement) groups.

    Args:
        dataframe: Full raw DataFrame from load_data().

    Returns:
        Summary DataFrame with one row per (provider, placement) group.
    """
    rows = []
    for (provider, placement), group in dataframe.groupby(["provider", "placement"]):
        stats = compute_group_stats(group)
        rows.append({"provider": provider, "placement": placement, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

# (metric_col, err_lo_col, err_hi_col, title, is_rate)
# For rates: err_lo/hi are Wilson CI bounds.
# For means: err_lo is SE column, err_hi is None (symmetric).
CHART_METRICS: list[tuple[str, str, str | None, str, bool]] = [
    ("completion_rate",  "completion_rate_lo", "completion_rate_hi", "Completion Rate",          True),
    ("compliance_rate",  "compliance_rate_lo", "compliance_rate_hi", "Compliance Rate",          True),
    ("mean_turns",       "se_turns",           None,                 "Mean Turns to Completion", False),
]


def _format_placement(name: str) -> str:
    """Convert snake_case placement name to Title Case for display."""
    return name.replace("_", " ").title()


def _all_same_or_nan(values: list[float]) -> bool:
    """Return True if all values are NaN or all identical — no useful variance."""
    real_values = [v for v in values if not math.isnan(v)]
    if not real_values:
        return True
    return max(real_values) - min(real_values) < 1e-9


def _bar_positions(n_providers: int, n_placements: int, provider_idx: int) -> list[float]:
    """Compute x-positions for a provider's bars in a grouped bar chart."""
    bar_width = 0.7 / max(n_providers, 1)
    offsets = [(i - (n_providers - 1) / 2) * bar_width for i in range(n_providers)]
    return [xi + offsets[provider_idx] for xi in range(n_placements)]


def save_chart(summary: pd.DataFrame) -> None:
    """Generate bar charts for metrics with meaningful variance and save to CHART_PATH.

    Automatically skips subplots where all values are identical (e.g. 100% completion
    rate, 1 turn per run) or all NaN (e.g. prefill_subsequent when every run is 1 turn).
    Annotates each bar with its value and uses Title Case placement labels.

    Args:
        summary: Summary DataFrame from build_summary().
    """
    providers = sorted(summary["provider"].unique().tolist())
    placements_order = sorted(summary["placement"].unique().tolist())
    placement_labels = [_format_placement(p) for p in placements_order]
    multi_provider = len(providers) > 1

    # Keep only metrics that have at least some variance across placements
    meaningful_metrics: list[tuple[str, str, str | None, str, bool]] = []
    for metric_tuple in CHART_METRICS:
        metric, _err_lo, _err_hi, _title, _is_rate = metric_tuple
        collected_values: list[float] = []
        for provider in providers:
            is_prefill = "prefill" in metric
            if is_prefill and provider != "ollama":
                continue
            provider_rows = summary[summary["provider"] == provider]
            for placement in placements_order:
                row = provider_rows[provider_rows["placement"] == placement]
                if not row.empty:
                    collected_values.append(float(row[metric].values[0]))
        if not _all_same_or_nan(collected_values):
            meaningful_metrics.append(metric_tuple)

    if not meaningful_metrics:
        logger.warning("No metrics with meaningful variance — skipping chart.")
        return

    n_charts = len(meaningful_metrics)
    n_cols = min(n_charts, 3)
    n_rows = math.ceil(n_charts / n_cols)

    fig, raw_axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = raw_axes.flatten()

    for ax_idx, (metric, err_lo_col, err_hi_col, title, is_rate) in enumerate(meaningful_metrics):
        ax = flat_axes[ax_idx]
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(placements_order)))
        ax.set_xticklabels(placement_labels, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        is_prefill = "prefill" in metric
        providers_for_chart = [pv for pv in providers if not is_prefill or pv == "ollama"]

        all_bar_tops: list[float] = []  # track max bar+err to set y-limit

        for provider_idx, provider in enumerate(providers_for_chart):
            provider_rows = summary[summary["provider"] == provider]
            values, err_below, err_above = [], [], []

            for placement in placements_order:
                row = provider_rows[provider_rows["placement"] == placement]
                if row.empty:
                    values.append(0.0)
                    err_below.append(0.0)
                    err_above.append(0.0)
                    continue

                val = float(row[metric].values[0])
                val = 0.0 if math.isnan(val) else val
                values.append(val)

                if is_rate:
                    lo = float(row[err_lo_col].values[0])
                    hi = float(row[err_hi_col].values[0])  # type: ignore[index]
                    err_below.append(0.0 if math.isnan(lo) else max(0.0, val - lo))
                    err_above.append(0.0 if math.isnan(hi) else max(0.0, hi - val))
                else:
                    se = float(row[err_lo_col].values[0])
                    err_sym = 0.0 if math.isnan(se) else se
                    err_below.append(err_sym)
                    err_above.append(err_sym)

            x_pos = _bar_positions(len(providers_for_chart), len(placements_order), provider_idx)
            bars = ax.bar(
                x_pos,
                values,
                width=0.7 / max(len(providers_for_chart), 1),
                label=provider,
                yerr=[err_below, err_above],
                capsize=4,
                color=f"C{provider_idx}",
            )

            # Annotate each bar with its value
            for bar, val, ea in zip(bars, values, err_above):
                label_text = f"{val:.0%}" if is_rate else f"{val:.0f}"
                bar_top = bar.get_height() + ea
                all_bar_tops.append(bar_top)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top,
                    label_text,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        if is_rate:
            # Give headroom for the % labels above the tallest bar+CI
            ax.set_ylim(0, min(1.0, max(all_bar_tops, default=1.0) * 1.25))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        else:
            ax.set_ylim(0, max(all_bar_tops, default=1.0) * 1.25)

        if multi_provider and len(providers_for_chart) > 1:
            ax.legend(fontsize=8)

    # Hide any unused subplot slots
    for unused_ax in flat_axes[n_charts:]:
        unused_ax.set_visible(False)

    providers_str = " + ".join(p.title() for p in providers)
    fig.suptitle(
        f"Prompt Placement Experiment — Results\n"
        f"Provider: {providers_str} | n=50 per placement",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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

    # Print table
    display_cols = [
        "provider", "placement", "total_runs", "completions",
        "completion_rate", "compliance_rate", "mean_turns", "mean_total_tokens",
        "mean_prefill_turn_1", "mean_prefill_subsequent",
    ]
    available_cols = [col for col in display_cols if col in summary.columns]
    print("\n" + summary[available_cols].round(3).to_string(index=False))
    print()

    save_chart(summary)
    logger.info("Analysis complete.")


def main() -> None:
    """Configure logging and run analysis."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    analyze()


if __name__ == "__main__":
    main()
