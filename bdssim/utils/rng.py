"""Seeded random number helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from numpy.random import Generator


@dataclass
class RNGManager:
    """Manage deterministic RNG streams."""

    seed: int
    _streams: Dict[str, Generator] = field(default_factory=dict, init=False)

    def generator(self, name: str) -> Generator:
        """Return a deterministic generator identified by ``name``."""

        if name not in self._streams:
            namespace = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
            seq = np.random.SeedSequence([self.seed, namespace & 0xFFFFFFFF, namespace >> 32])
            self._streams[name] = np.random.default_rng(seq)
        return self._streams[name]

    def reset(self) -> None:
        """Clear cached generators."""

        self._streams.clear()


__all__ = ["RNGManager"]
