"""Bass diffusion model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from numpy.random import Generator

from bdssim.config import AdoptionBassParams


@dataclass
class BassState:
    params: AdoptionBassParams
    candidates: Iterable[str]
    stochastic: bool = False
    rng: Generator | None = None
    adopters: List[str] = field(default_factory=list)
    day: int = 0
    _carry: float = 0.0

    @property
    def adopters_count(self) -> int:
        return len(self.adopters)

    def step(self, dt: int = 1, allowed: Iterable[str] | None = None) -> List[str]:
        allowed_set = set(allowed) if allowed is not None else None
        new_adopters: List[str] = []
        m = self.params.m
        for _ in range(dt):
            if self.adopters_count >= m:
                self.day += 1
                continue
            remaining = [c for c in self.candidates if c not in self.adopters]
            if allowed_set is not None:
                eligible = [c for c in remaining if c in allowed_set]
            else:
                eligible = remaining
            if not eligible:
                self.day += 1
                continue
            share = self.adopters_count / m
            expectation = (self.params.p + self.params.q * share) * (m - self.adopters_count)
            if self.stochastic and self.rng is not None:
                draw = int(self.rng.poisson(max(expectation, 0)))
            else:
                self._carry += expectation
                draw = int(self._carry)
                self._carry -= draw
            draw = max(min(draw, len(eligible), m - self.adopters_count), 0)
            if draw <= 0:
                self.day += 1
                continue
            selection = eligible[:draw]
            for code in selection:
                if code not in self.adopters:
                    self.adopters.append(code)
                    new_adopters.append(code)
            self.day += 1
        return new_adopters


__all__ = ["BassState"]
