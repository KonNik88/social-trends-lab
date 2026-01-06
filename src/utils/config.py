from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @staticmethod
    def detect() -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        return ProjectPaths(root=root)

    def abs(self, rel: str) -> Path:
        return (self.root / rel).resolve()


class Cfg:
    """Tiny helper for dot-ish access: cfg.get('a.b.c', default)."""

    def __init__(self, d: Dict[str, Any]):
        self._d = d

    def get(self, key: str, default=None):
        cur: Any = self._d
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def as_dict(self) -> Dict[str, Any]:
        return self._d
