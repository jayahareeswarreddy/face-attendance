"""Runtime configuration.

Every tunable lives here so behaviour can be changed without touching logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("FACEATTENDANCE_DATA", "data")).expanduser()


@dataclass(frozen=True)
class Config:
    """Paths and thresholds for a single run."""

    data_dir: Path
    camera_index: int = 0
    samples_per_student: int = 25
    face_size: tuple[int, int] = (200, 200)
    warmup_frames: int = 12
    # LBPH returns a DISTANCE: lower is a better match. Anything above this
    # is treated as an unknown face rather than a weak guess.
    max_distance: float = 70.0

    @classmethod
    def load(cls, data_dir: Path | None = None, **overrides: object) -> "Config":
        return cls(data_dir=data_dir or _default_data_dir(), **overrides)  # type: ignore[arg-type]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "attendance.db"

    @property
    def model_path(self) -> Path:
        return self.data_dir / "model.yml"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    def ensure_dirs(self) -> None:
        self.faces_dir.mkdir(parents=True, exist_ok=True)
