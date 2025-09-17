"""Markdown reporting for scenario Monte Carlo results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from bdssim.config import Config
from bdssim.playbooks import Playbook, get_playbook_by_key
from bdssim.reporting.debt import build_debt_projection
from bdssim.reporting.static_meta import ScenarioStaticMeta, scenario_meta_for


@dataclass
class ScenarioMCReport:
    name: str
    description: str | None
    metrics: pd.DataFrame
    percentiles: pd.DataFrame
    timepoint_percentiles: pd.DataFrame
    config: Config
    metadata: ScenarioStaticMeta | None
    playbook: Playbook | None

    @property
    def median_row(self) -> pd.Series:
        return self.percentiles.set_index("percentile").loc[0.5]


_CHECKPOINT_YEARS = (2, 5, 7)


def _format_percentiles_table(percentiles: pd.DataFrame) -> str:
    df = percentiles.copy()
    df["percentile"] = (df["percentile"] * 100).astype(int)
    df.rename(columns={"percentile": "Percentile"}, inplace=True)
    return df.to_markdown(index=False, floatfmt=".4f")


def _scenario_context_section(result: ScenarioMCReport) -> list[str]:
    meta = result.metadata
    lines: list[str] = []
    if meta:
        lines.extend(
            [
                "### Strategic context",
                meta.summary,
                "",
                f"**Confidence:** {meta.confidence}",
                f"**Debt growth anchor:** {meta.debt_source} (CAGR {meta.debt_cagr*100:.2f}%)",
            ]
        )
        if meta.key_assumptions:
            lines.append("**Key assumptions:**")
            for assumption in meta.key_assumptions:
                lines.append(f"- {assumption}")
        lines.append("")
    return lines


def _playbook_section(result: ScenarioMCReport) -> list[str]:
    playbook = result.playbook
    if playbook is None:
        return []
    rows = []
    for wave in playbook.waves:
        name = wave.get("name", "Stage")
        start_day = wave.get("start_day", 0)
        countries = ", ".join(wave.get("countries", []))
        blocs = ", ".join(wave.get("blocs", [])) if wave.get("blocs") else "—"
        probability = wave.get("probability")
        prob_str = f"{probability:.0%}" if isinstance(probability, (int, float)) else "n/a"
        rows.append({
            "Stage": name,
            "Start Day": start_day,
            "Countries": countries,
            "Blocs": blocs,
            "Stage Prob": prob_str,
        })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    lines = ["### Adoption wave schedule", df.to_markdown(index=False)]
    if playbook.sources:
        lines.append("")
        lines.append("**Sources:** " + "; ".join(playbook.sources))
    if playbook.assumptions:
        lines.append("")
        lines.append(f"**Assumptions:** {playbook.assumptions}")
    lines.append("")
    return lines


def _checkpoint_table(result: ScenarioMCReport) -> list[str]:
    if result.timepoint_percentiles.empty:
        return []
    baseline = result.config.objective.debt_baseline_usd
    meta = result.metadata
    cagr = meta.debt_cagr if meta else 0.0
    projections = {row.years: row.projected_debt for row in build_debt_projection(baseline, cagr, _CHECKPOINT_YEARS)}

    records = []
    df = result.timepoint_percentiles
    for years in sorted(df["years"].unique()):
        slice_df = df[df["years"] == years]
        projected = projections.get(years, baseline)
        for _, row in slice_df.iterrows():
            percentile = int(row["percentile"] * 100)
            holdings_btc = row["us_holdings_btc"]
            price = row["price"]
            holdings_value = holdings_btc * price
            coverage_on_baseline = row["debt_coverage"] * 100
            coverage_on_projected = (holdings_value / projected * 100) if projected else float("nan")
            records.append(
                {
                    "Years": years,
                    "Percentile": percentile,
                    "Holdings BTC": holdings_btc,
                    "Price USD": price,
                    "Holdings USD": holdings_value,
                    "Projected Debt USD": projected,
                    "Coverage vs Baseline %": coverage_on_baseline,
                    "Coverage vs Projected %": coverage_on_projected,
                }
            )
    checkpoint_df = pd.DataFrame(records)
    if checkpoint_df.empty:
        return []
    display_df = checkpoint_df.copy()
    for col in ["Holdings BTC", "Price USD", "Holdings USD", "Projected Debt USD"]:
        display_df[col] = display_df[col].map(lambda val: f"{val:,.0f}")
    for col in ["Coverage vs Baseline %", "Coverage vs Projected %"]:
        display_df[col] = display_df[col].map(lambda val: f"{val:.2f}")
    lines = ["### Holdings vs debt checkpoints", display_df.to_markdown(index=False)]
    lines.append("")
    return lines


def _scenario_section(result: ScenarioMCReport) -> str:
    median = result.median_row
    md = [f"## Scenario: {result.name}"]
    if result.description:
        md.append(result.description)
    md.append("")
    md.extend(_scenario_context_section(result))
    md.extend(
        [
            "**Median Outcomes (p50)**",
            f"- Final price: ${median['final_price']:,.0f}",
            f"- Debt coverage: {median['final_coverage']*100:.2f}%",
            f"- Peak price: ${median.get('peak_price', float('nan')):,.0f}",
            f"- Max drawdown: {median.get('max_drawdown', float('nan')):.2%}",
            f"- U.S. holdings: {median.get('us_holdings_btc', float('nan')):,.0f} BTC",
            f"- Sovereign adopters: {median.get('adopters_final', float('nan')):,.0f}",
            "",
            "**Percentile Table (p5/p50/p95)**",
            _format_percentiles_table(result.percentiles),
            "",
        ]
    )
    md.extend(_checkpoint_table(result))
    md.extend(_playbook_section(result))
    return "\n".join(md)


def build_markdown_report(results: Sequence[ScenarioMCReport]) -> str:
    intro = [
        "# Bitcoin Debt Solution Monte Carlo Report",
        "",
        "This report aggregates scenario outcomes using Monte Carlo simulations.",
        "Each scenario shows key percentiles across the simulated distribution of outcomes.",
        "",
    ]
    comparison_section: list[str] = []
    if len(results) >= 2:
        comparison_section.append("## Scenario Comparison (Median)")
        rows = []
        for result in results:
            median = result.median_row
            rows.append(
                {
                    "Scenario": result.name,
                    "Final Price": median.get("final_price", float("nan")),
                    "Debt Coverage %": median.get("final_coverage", float("nan")) * 100,
                    "Peak Price": median.get("peak_price", float("nan")),
                    "Max Drawdown": median.get("max_drawdown", float("nan")),
                    "Holdings BTC": median.get("us_holdings_btc", float("nan")),
                    "Adopters": median.get("adopters_final", float("nan")),
                }
            )
        comp_df = pd.DataFrame(rows)
        comparison_section.append(comp_df.to_markdown(index=False, floatfmt=".2f"))
        comparison_section.append("")
    body = [_scenario_section(result) for result in results]
    return "\n".join(intro + comparison_section + body)


def save_report(markdown: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")


def scenario_report_from_mc(
    *,
    config: Config,
    metrics: pd.DataFrame,
    percentiles: pd.DataFrame,
    timepoint_percentiles: pd.DataFrame,
    playbook: Playbook | None = None,
) -> ScenarioMCReport:
    metadata = scenario_meta_for(config.meta.name)
    if metadata and playbook is None and metadata.playbook_key:
        playbook = get_playbook_by_key(metadata.playbook_key)
    return ScenarioMCReport(
        name=config.meta.name,
        description=config.meta.description,
        metrics=metrics,
        percentiles=percentiles,
        timepoint_percentiles=timepoint_percentiles,
        config=config,
        metadata=metadata,
        playbook=playbook,
    )


__all__ = [
    "ScenarioMCReport",
    "build_markdown_report",
    "save_report",
    "scenario_report_from_mc",
]

