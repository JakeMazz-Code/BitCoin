"""Playbook loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "configs" / "playbooks" / "index.yaml"


@dataclass
class Playbook:
    key: str
    name: str
    description: str
    probability: float | None
    waves: List[dict]
    confidence: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    assumptions: Optional[str] = None


def _coerce_playbook_entry(entry: dict) -> Playbook:
    return Playbook(
        key=str(entry.get("key", entry.get("name", "playbook"))),
        name=str(entry.get("name", entry.get("key", "Playbook"))),
        description=str(entry.get("description", "")),
        probability=entry.get("probability"),
        waves=list(entry.get("waves", [])),
        confidence=entry.get("confidence"),
        sources=[str(src) for src in entry.get("sources", []) if src],
        assumptions=entry.get("assumptions"),
    )


def load_playbook_index(path: str | Path | None = None) -> List[Playbook]:
    """Return playbook metadata from the curated index.

    Parameters
    ----------
    path:
        Optional path to an index YAML file. Defaults to the repository-level
        ``configs/playbooks/index.yaml``.
    """

    path_obj = Path(path) if path is not None else DEFAULT_INDEX_PATH
    if not path_obj.exists():
        raise FileNotFoundError(f"Playbook index not found at {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    entries = data.get("playbooks", data)
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list):
        raise ValueError("Playbook index must define a list of playbooks")

    playbooks: List[Playbook] = []
    for entry in entries:
        if isinstance(entry, dict):
            playbooks.append(_coerce_playbook_entry(entry))
    return playbooks


def get_playbook_by_key(key: str, index_path: str | Path | None = None) -> Optional[Playbook]:
    key_lower = key.lower()
    for playbook in load_playbook_index(index_path):
        if playbook.key.lower() == key_lower:
            return playbook
    return None


def reduce_playbook(playbook: Playbook) -> List[dict]:
    """Return list of wave dictionaries suitable for SimulationEngine."""

    waves: List[dict] = []
    for wave in playbook.waves:
        if not isinstance(wave, dict):
            continue
        name = wave.get("name", "Stage")
        start_day = int(wave.get("start_day", 0))
        countries = list(wave.get("countries", []))
        blocs = list(wave.get("blocs", []))
        if not countries:
            continue
        stage = {"name": name, "start_day": start_day, "countries": countries}
        if blocs:
            stage["blocs"] = blocs
        if "probability" in wave:
            stage["probability"] = wave["probability"]
        waves.append(stage)
    return waves


__all__ = ["Playbook", "load_playbook_index", "get_playbook_by_key", "reduce_playbook"]
