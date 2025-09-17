"""Command line interface for bdssim."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import typer
import yaml
from rich.console import Console

from bdssim.analysis import (
    compute_network_centrality,
    greedy_seed_selection,
    price_distribution,
    summarize_cascade,
)
from bdssim.config import Config, load_config
from bdssim.data.market_data import fetch_spot_price
from bdssim.engine.monte_carlo import run_mc
from bdssim.engine.simulation import SimulationEngine
from bdssim.playbooks import get_playbook_by_key, load_playbook_index, reduce_playbook
from bdssim.reporting.pdf import markdown_to_pdf
from bdssim.reporting.plots import plot_all
from bdssim.reporting.report import build_markdown_report, save_report, scenario_report_from_mc
from bdssim.reporting.static_meta import scenario_meta_for
from bdssim.reporting.summary import export_summary
from bdssim.utils.io import load_result_tables, timestamped_dir
from bdssim.utils.validation import validate_config

app = typer.Typer(help="Bitcoin debt solution simulation CLI")
console = Console()


def _resolve_output(base: Path, scenario: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return timestamped_dir(base, scenario)


def _with_live_price(cfg: Config, use_live_price: bool) -> Config:
    if not use_live_price:
        return cfg
    price = fetch_spot_price()
    console.print(f"[bold cyan]Fetched live BTC price[/bold cyan]: ${price:,.0f}")
    market = cfg.market.model_copy(update={"init_price_usd": price})
    return cfg.model_copy(update={"market": market})


def _load_playbook_file(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, dict) and "playbook" in data:
        entries = data["playbook"]
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError(f"Playbook file {path} must contain a list of stages")
    normalized: List[dict[str, Any]] = []
    for stage in entries:
        if not isinstance(stage, dict):
            raise ValueError(f"Invalid stage definition: {stage!r}")
        countries = list(stage.get("countries", []))
        normalized.append(
            {
                "name": stage.get("name", "Stage"),
                "start_day": int(stage.get("start_day", 0)),
                "countries": countries,
            }
        )
    return normalized


def _resolve_playbook(
    cfg: Config, playbook_path: Optional[Path], playbook_name: Optional[str]
) -> List[dict[str, Any]]:
    if playbook_name:
        playbook = get_playbook_by_key(playbook_name)
        if playbook is None:
            available = ", ".join(pb.key for pb in load_playbook_index())
            raise ValueError(f"Unknown playbook '{playbook_name}'. Available: {available}")
        return reduce_playbook(playbook)
    if playbook_path is not None:
        return _load_playbook_file(playbook_path)
    if cfg.playbook:
        return [stage.model_dump() for stage in cfg.playbook]
    return []


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, help="Configuration YAML"),
    out: Path = typer.Option(Path("results/base"), help="Output directory"),
    data_root: Path = typer.Option(Path("."), help="Data directory base"),
    use_live_price: bool = typer.Option(False, help="Fetch current BTC/USD spot price"),
    playbook: Optional[Path] = typer.Option(None, help="Playbook YAML with staged adoption", exists=True, dir_okay=False, file_okay=True),
    playbook_name: Optional[str] = typer.Option(None, help="Named playbook key from configs/playbooks/index.yaml"),
) -> None:
    """Run a single deterministic simulation."""

    cfg = _with_live_price(load_config(config), use_live_price)
    validate_config(cfg)
    scenario_name = cfg.meta.name or config.stem
    out_dir = _resolve_output(out, scenario_name)
    console.print(f"[bold green]Running simulation[/bold green] -> {out_dir}")
    playbook_data = _resolve_playbook(cfg, playbook, playbook_name)
    engine = SimulationEngine(cfg, data_root=data_root, playbook=playbook_data)
    result = engine.run(out_dir=out_dir)
    export_summary(result.timeseries, out_dir)
    plot_all(result.timeseries, out_dir, scenario_name)
    console.print("Simulation complete")


@app.command()
def mc(
    config: Path = typer.Option(..., exists=True, help="Configuration YAML"),
    runs: int = typer.Option(100, min=1, help="Number of Monte Carlo runs"),
    out: Path = typer.Option(Path("results/mc"), help="Output directory"),
    data_root: Path = typer.Option(Path("."), help="Data directory"),
    use_live_price: bool = typer.Option(False, help="Fetch current BTC/USD spot price"),
    playbook: Optional[Path] = typer.Option(None, help="Playbook YAML with staged adoption", exists=True, dir_okay=False, file_okay=True),
    playbook_name: Optional[str] = typer.Option(None, help="Named playbook key from configs/playbooks/index.yaml"),
) -> None:
    """Run Monte Carlo study and persist aggregate metrics."""

    cfg = _with_live_price(load_config(config), use_live_price)
    validate_config(cfg)
    playbook_data = _resolve_playbook(cfg, playbook, playbook_name)
    out_dir = _resolve_output(out, cfg.meta.name or config.stem)
    console.print(f"[bold cyan]Running Monte Carlo[/bold cyan] -> {out_dir}")
    result = run_mc(cfg, runs=runs, data_root=data_root, out_dir=out_dir, playbook=playbook_data)
    console.print(result.percentiles)


@app.command()
def compare(
    configs: List[Path] = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("results/compare"), help="Output directory"),
    data_root: Path = typer.Option(Path("."), help="Data directory"),
    use_live_price: bool = typer.Option(False, help="Fetch current BTC/USD spot price"),
    playbook: Optional[Path] = typer.Option(None, help="Playbook YAML with staged adoption", exists=True, dir_okay=False, file_okay=True),
    playbook_name: Optional[str] = typer.Option(None, help="Named playbook key from configs/playbooks/index.yaml"),
) -> None:
    """Run multiple configurations and collate summaries."""

    out.mkdir(parents=True, exist_ok=True)
    live_price: Optional[float] = None
    if use_live_price:
        live_price = fetch_spot_price()
        console.print(f"[bold cyan]Fetched live BTC price[/bold cyan]: ${live_price:,.0f}")
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        if live_price is not None:
            cfg = cfg.model_copy(update={"market": cfg.market.model_copy(update={"init_price_usd": live_price})})
        validate_config(cfg)
        playbook_data = _resolve_playbook(cfg, playbook, playbook_name)
        scenario_dir = _resolve_output(out, cfg.meta.name or cfg_path.stem)
        engine = SimulationEngine(cfg, data_root=data_root, playbook=playbook_data)
        result = engine.run(out_dir=scenario_dir)
        export_summary(result.timeseries, scenario_dir)
        plot_all(result.timeseries, scenario_dir, cfg.meta.name or cfg_path.stem)
        console.print(f"[bold]{cfg.meta.name or cfg_path.stem}[/bold] final coverage {result.timeseries['debt_coverage'].iloc[-1]:.4%}")


@app.command()
def plot(
    result: Path = typer.Option(..., exists=True, file_okay=False, help="Result directory"),
    what: Optional[List[str]] = typer.Argument(None),
) -> None:
    """Regenerate plots from stored result tables."""

    tables = load_result_tables(result)
    ts = tables["timeseries"]
    generated = plot_all(ts, result, result.name)
    selections = list(what) if what else ["price", "adoption", "coverage"]
    name_map = {"price": 0, "adoption": 1, "coverage": 2}
    for item in selections:
        idx = name_map.get(item)
        if idx is None:
            console.print(f"Unknown plot request {item}")
            continue
        console.print(f"Saved {generated[idx]}")


@app.command()
def analyze(
    config: Path = typer.Option(..., exists=True, help="Configuration YAML"),
    out: Path = typer.Option(Path("results/analysis"), help="Output directory"),
    data_root: Path = typer.Option(Path("."), help="Data directory"),
    top_k: int = typer.Option(5, min=1, max=10, help="Top-N countries to highlight"),
    use_live_price: bool = typer.Option(False, help="Fetch current BTC/USD spot price"),
    playbook: Optional[Path] = typer.Option(None, help="Playbook YAML with staged adoption", exists=True, dir_okay=False, file_okay=True),
    playbook_name: Optional[str] = typer.Option(None, help="Named playbook key from configs/playbooks/index.yaml"),
) -> None:
    """Run a scenario and output advanced analytics (network centrality, cascade stats, seeds)."""

    cfg = _with_live_price(load_config(config), use_live_price)
    validate_config(cfg)
    playbook_data = _resolve_playbook(cfg, playbook, playbook_name)
    scenario_name = cfg.meta.name or config.stem
    out_dir = _resolve_output(out, scenario_name)
    console.print(f"[bold yellow]Analyzing scenario[/bold yellow] -> {out_dir}")
    engine = SimulationEngine(cfg, data_root=data_root, playbook=playbook_data)
    result = engine.run(out_dir=out_dir)

    centrality = compute_network_centrality(engine.catalog)
    cascade = summarize_cascade(result.timeseries)
    seeds = greedy_seed_selection(engine.catalog, k=top_k)
    price_stats = price_distribution(result.timeseries)

    centrality_path = out_dir / "centrality_ranking.csv"
    centrality.to_csv(centrality_path, index=False)

    adoption_log = pd.DataFrame(engine.adoption_log, columns=["day", "country"])
    adoption_path = out_dir / "adoption_log.csv"
    adoption_log.to_csv(adoption_path, index=False)

    markdown_lines = [
        f"# Scenario Analytics: {scenario_name}",
        "",
        cfg.meta.description or "",
        "",
        "## Price Diagnostics",
        price_stats.to_frame(name="value").to_markdown(),
        "",
        "## Adoption Cascade",
        f"- Total adopters: {cascade.total_adopters}",
        f"- Adoption rate (per day): {cascade.adoption_rate:.3f}",
        f"- Time to half adoption: {cascade.time_to_half if cascade.time_to_half is not None else 'N/A'} days",
        "",
        "## Recommended Seed Countries",
        "- " + ", ".join(seeds),
        "",
        f"## Network Centrality (top {top_k})",
        centrality.head(top_k).to_markdown(index=False),
        "",
        "## Adoption Timeline",
        adoption_log.head(20).to_markdown(index=False),
        "",
        "Artifacts generated:",
        f"- Centrality ranking: {centrality_path}",
        f"- Adoption log: {adoption_path}",
    ]

    report_path = out_dir / "analysis.md"
    report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    console.print(f"Saved analytics to {report_path}")


@app.command()
def report(
    configs: List[Path] = typer.Argument(..., exists=True),
    runs: int = typer.Option(500, min=10, help="Monte Carlo runs per scenario"),
    out: Path = typer.Option(Path("results/reports"), help="Directory for markdown report"),
    data_root: Path = typer.Option(Path("."), help="Data directory"),
    use_live_price: bool = typer.Option(False, help="Fetch current BTC/USD spot price"),
    as_pdf: bool = typer.Option(False, help="Also render the report to PDF"),
    playbook: Optional[Path] = typer.Option(None, help="Playbook YAML with staged adoption", exists=True, dir_okay=False, file_okay=True),
    playbook_name: Optional[str] = typer.Option(None, help="Named playbook key from configs/playbooks/index.yaml"),
) -> None:
    """Generate a Markdown report summarising Monte Carlo outcomes for scenarios."""

    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    live_price: Optional[float] = None
    if use_live_price:
        live_price = fetch_spot_price()
        console.print(f"[bold cyan]Fetched live BTC price[/bold cyan]: ${live_price:,.0f}")
    results = []
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        if live_price is not None:
            cfg = cfg.model_copy(update={"market": cfg.market.model_copy(update={"init_price_usd": live_price})})
        validate_config(cfg)
        playbook_data = _resolve_playbook(cfg, playbook, playbook_name)
        console.print(f"[bold magenta]Running MC for {cfg.meta.name or cfg_path.stem} ({runs} runs)...")
        mc_result = run_mc(cfg, runs=runs, data_root=data_root, playbook=playbook_data)
        static_meta = scenario_meta_for(cfg.meta.name or cfg_path.stem)
        playbook_obj = None
        if playbook_name:
            playbook_obj = get_playbook_by_key(playbook_name)
        elif static_meta and static_meta.playbook_key:
            playbook_obj = get_playbook_by_key(static_meta.playbook_key)
        results.append(
            scenario_report_from_mc(
                config=cfg,
                metrics=mc_result.metrics,
                percentiles=mc_result.percentiles,
                timepoint_percentiles=mc_result.timepoint_percentiles,
                playbook=playbook_obj,
            )
        )
    markdown = build_markdown_report(results)
    target = out / f"bdssim_report_{timestamp}.md"
    save_report(markdown, target)
    console.print(f"Saved report to {target}")
    if as_pdf:
        pdf_path = target.with_suffix(".pdf")
        markdown_to_pdf(markdown, pdf_path)
        console.print(f"Saved PDF report to {pdf_path}")




@app.command()
def playbooks(
    index: Path = typer.Option(Path("configs/playbooks/index.yaml"), help="Playbook index YAML")
) -> None:
    """List available staged-adoption playbooks."""

    console.print("Available playbooks:\n")
    for playbook in load_playbook_index(index):
        prob = f"{playbook.probability:.0%}" if playbook.probability is not None else "n/a"
        console.print(f"[bold]{playbook.key}[/bold] ({prob}) - {playbook.name}")
        if playbook.description:
            console.print(f"  {playbook.description}")
        console.print("")
@app.command()
def validate(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Validate configuration without running simulation."""

    cfg = load_config(config)
    validate_config(cfg)
    console.print("Configuration validated successfully")


if __name__ == "__main__":
    app()



