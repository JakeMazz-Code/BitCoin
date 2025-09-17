"""Bitcoin debt solution simulation package."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("bitcoin-debt-solution-sim")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for local usage
        return "0.1.0"


__all__ = ["get_version"]
