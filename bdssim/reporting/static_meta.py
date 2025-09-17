"""Static scenario metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

STATIC_META_PATH = Path(__file__).resolve().parents[2] / "configs" / "static" / "metadata.yaml"


@dataclass(frozen=True)
class ScenarioStaticMeta:
    key: str
    debt_cagr: float
    debt_source: str
    summary: str
    confidence: str
    key_assumptions: tuple[str, ...]
    playbook_key: Optional[str] = None


_cache: Dict[str, ScenarioStaticMeta] | None = None


def _load_raw(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Static metadata file must contain a mapping")
    return data.get("scenarios", data)


def load_static_metadata(path: Path | None = None) -> Dict[str, ScenarioStaticMeta]:
    global _cache
    if _cache is not None and path is None:
        return _cache
    target = path or STATIC_META_PATH
    raw = _load_raw(target)
    result: Dict[str, ScenarioStaticMeta] = {}
    for key, values in raw.items():
        if not isinstance(values, dict):
            continue
        assumptions = tuple(str(item) for item in values.get("key_assumptions", []) if item)
        meta = ScenarioStaticMeta(
            key=key,
            debt_cagr=float(values.get("debt_cagr", 0.0)),
            debt_source=str(values.get("debt_source", "")),
            summary=str(values.get("summary", "")),
            confidence=str(values.get("confidence", "")),
            key_assumptions=assumptions,
            playbook_key=values.get("playbook_key"),
        )
        result[key] = meta
    if path is None:
        _cache = result
    return result


def scenario_meta_for(name: str) -> Optional[ScenarioStaticMeta]:
    metadata = load_static_metadata()
    return metadata.get(name)


__all__ = ["ScenarioStaticMeta", "load_static_metadata", "scenario_meta_for"]
