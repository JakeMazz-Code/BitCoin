"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_price(timeseries: pd.DataFrame, out_dir: Path, scenario: str) -> Path:
    out_dir = _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeseries["day"], timeseries["price"], label="Price")
    ax.set_title(f"BTC Price Path - {scenario}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    path = out_dir / f"{scenario}_price.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_adoption(timeseries: pd.DataFrame, out_dir: Path, scenario: str) -> Path:
    out_dir = _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeseries["day"], timeseries["adopters"], label="Adopters", color="tab:orange")
    ax.set_title(f"Sovereign Adoption - {scenario}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Countries Adopting")
    ax.legend()
    path = out_dir / f"{scenario}_adoption.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_debt_coverage(timeseries: pd.DataFrame, out_dir: Path, scenario: str) -> Path:
    out_dir = _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeseries["day"], timeseries["debt_coverage"], label="Debt Coverage", color="tab:green")
    ax.set_title(f"Debt Coverage Ratio - {scenario}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Coverage (fraction)")
    ax.legend()
    path = out_dir / f"{scenario}_coverage.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_all(timeseries: pd.DataFrame, out_dir: Path, scenario: str) -> list[Path]:
    return [
        plot_price(timeseries, out_dir, scenario),
        plot_adoption(timeseries, out_dir, scenario),
        plot_debt_coverage(timeseries, out_dir, scenario),
    ]


__all__ = ["plot_price", "plot_adoption", "plot_debt_coverage", "plot_all"]
