"""Markdown builder for post-cap monetization reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from bdssim.ceilings import COFERInputs, TAMInputs, cofer_share_ceiling, tam_share_ceiling


def _df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in headers) + " |" for _, row in df.iterrows()]
    return "\n".join([header_line, separator] + body)


@dataclass
class CeilingInputs:
    tam: Optional[TAMInputs]
    cofer: Optional[COFERInputs]


def _plot_postcap_bend(sim_df: pd.DataFrame, switch_year: int, out_path: Path) -> None:
    if "day" in sim_df.columns:
        x = sim_df["day"].to_numpy()
        x_label = "Day"
        switch_line = switch_year * 365
    else:
        x = sim_df["year"].to_numpy()
        x_label = "Year"
        switch_line = float(switch_year)
    y = sim_df["price"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, label="Price path", color="tab:blue")
    ax.axvline(switch_line, color="tab:red", linestyle="--", label="Post-cap switch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Price (USD)")
    ax.set_title("Scarcity bend across the post-cap switch")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _format_currency(value: float) -> str:
    return f"${value:,.0f}" if pd.notna(value) else "n/a"


def _extract_ceiling_inputs(meta: Dict[str, Any]) -> CeilingInputs:
    tam_raw = meta.get("tam_inputs")
    cofer_raw = meta.get("cofer_inputs")
    tam = None
    cofer = None
    if tam_raw:
        tam = TAMInputs(
            tam_usd=float(tam_raw.get("tam_usd", 0.0)),
            btc_share=float(tam_raw.get("btc_share", 0.0)),
            tradable_supply=float(tam_raw.get("tradable_supply", 0.0)),
        )
    if cofer_raw:
        cofer = COFERInputs(
            cofer_usd=float(cofer_raw.get("cofer_usd", 0.0)),
            reserve_share=float(cofer_raw.get("reserve_share", 0.0)),
            tradable_supply=float(cofer_raw.get("tradable_supply", 0.0)),
        )
    return CeilingInputs(tam=tam, cofer=cofer)


def build_markdown_report(
    *,
    sim_df: pd.DataFrame,
    meta: Dict[str, Any],
    ceilings_df: pd.DataFrame,
    out_dir: Path,
    filename: str = "post_cap_report.md",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    switch_year_index = int(meta.get("postcap_year_index", 0))
    alpha_delta = float(meta.get("alpha_postcap_delta", 0.0))
    demand_delta = float(meta.get("demand_growth_postcap_delta", 0.0))
    switch_sentence = (
        f"After year {switch_year_index}, BTC is like gold - new supply is basically fixed. "
        "We nudge the model's scarcity curve (alpha) and demand growth (g) so price can still rise as more people want it."
    )

    chart_path = out_dir / "postcap_bend.png"
    _plot_postcap_bend(sim_df, switch_year_index, chart_path)

    last_row = sim_df.iloc[-1]
    observability: list[str] = []
    prior = last_row.get("prior_year_return")
    if pd.notna(prior):
        observability.append(f"Latest prior-year return: {prior:.1%}")
    else:
        observability.append("Prior-year return not available")
    reflex = last_row.get("reflexivity_delta")
    if pd.notna(reflex):
        observability.append(f"Reflexivity parked this year: {reflex:,.0f} BTC")
    else:
        observability.append("Reflexivity delta unavailable")

    table_df = ceilings_df.copy()
    if "value" in table_df.columns:
        table_df["value"] = table_df["value"].apply(_format_currency)
    for col in ("p10", "p50", "p90"):
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(_format_currency)
    table_df = table_df.rename(columns={
        "metric": "Metric",
        "value": "Implied price",
        "p10": "Model P10",
        "p50": "Model P50",
        "p90": "Model P90",
    })
    ceilings_table = _df_to_markdown(table_df)

    inputs = _extract_ceiling_inputs(meta)
    input_lines: list[str] = []
    if inputs.tam:
        implied = tam_share_ceiling(inputs.tam)
        input_lines.append(
            f"- TAM lens: {inputs.tam.btc_share:.0%} of ${inputs.tam.tam_usd/1e12:.1f}T -> {_format_currency(implied)}"
        )
    if inputs.cofer:
        implied = cofer_share_ceiling(inputs.cofer)
        input_lines.append(
            f"- COFER lens: {inputs.cofer.reserve_share:.0%} of ${inputs.cofer.cofer_usd/1e12:.1f}T -> {_format_currency(implied)}"
        )

    lines = [
        "# Post-Cap Monetization",
        "",
        switch_sentence,
        "",
        f"- Switch year index: {switch_year_index}",
        f"- Delta alpha after switch: {alpha_delta:+.2f}",
        f"- Delta g after switch: {demand_delta:+.2f}",
        "",
        "![Post-Cap Bend](postcap_bend.png)",
        "",
        "Key observability cues:",
    ]
    lines.extend(observability)
    lines.extend([
        "",
        "# How High Could It Go?",
        "",
        "These are not forecasts - just implied prices given the inputs.",
        "",
        ceilings_table,
    ])

    if input_lines:
        lines.extend(["", "Inputs used:"] + input_lines)

    horizon_price = float(last_row.get("price", float("nan")))
    lines.extend(["", f"Model horizon price (latest run): {_format_currency(horizon_price)}"])

    report_path = out_dir / filename
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


__all__ = ["build_markdown_report"]
