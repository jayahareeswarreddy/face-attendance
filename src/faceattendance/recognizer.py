"""LBPH face recognition.

This is the part the original prototype did not have. Detecting that *a*
face is present proves only that somebody is sitting in front of the
camera; it says nothing about who. Here the label comes out of the model,
and the caller never gets to assert an identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class ModelNotTrained(RuntimeError):
    """No trained model exists on disk yet."""


@dataclass(frozen=True)
class Match:
    """A recognition result.

    `distance` is the LBPH confidence score, where LOWER means a closer
    match. `accepted` is False when the distance exceeded the configured
    threshold, i.e. the face belongs to nobody we know.
    """

    label: int
    distance: float
    accepted: bool


class FaceRecognizer:
    """Thin wrapper over cv2.face.LBPHFaceRecognizer.

    LBPH is used rather than an embedding model because it trains on a
    couple of dozen samples per person, needs no GPU, and ships with
    opencv-contrib. Swapping it for an embedding-based recogniser means
    reimplementing this class and nothing else.
    """

    def __init__(self, max_distance: float = 70.0) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is unavailable - install opencv-contrib-python "
                "(the plain opencv-python wheel omits the contrib modules)"
            )
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._max_distance = max_distance
        self._trained = False

    def train(self, faces: list[np.ndarray], labels: list[int]) -> None:
        if not faces:
            raise ValueError("cannot train on an empty sample set")
        if len(faces) != len(labels):
            raise ValueError("faces and labels must be the same length")
        if len(set(labels)) < 2:
            # LBPH will happily "train" on one identity and then match
            # every face in the world to it. Refuse rather than mislead.
            raise ValueError(
                "at least two enrolled students are required before training; "
                "a single-identity model matches everyone"
            )
        self._model.train(np.array(faces, dtype="uint8"), np.array(labels, dtype=np.int32))
        self._trained = True

    def predict(self, face: np.ndarray) -> Match:
        if not self._trained:
            raise ModelNotTrained("train or load a model before predicting")
        label, distance = self._model.predict(face)
        return Match(int(label), float(distance), distance <= self._max_distance)

    def save(self, path: Path) -> None:
        if not self._trained:
            raise ModelNotTrained("nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.write(str(path))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise ModelNotTrained(
                f"no model at {path} — run 'faceattendance train' first"
            )
        self._model.read(str(path))
        self._trained = True
