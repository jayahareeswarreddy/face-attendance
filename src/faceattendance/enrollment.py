"""Enrolment: capturing face samples and training the recogniser."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera import open_camera, read_frame
from .config import Config
from .detector import FaceDetector
from .recognizer import FaceRecognizer
from . import storage

Progress = Callable[[int, int], None]


def capture_samples(
    config: Config,
    label: int,
    on_progress: Progress | None = None,
    show_preview: bool = True,
) -> int:
    """Capture face samples for one student and write them to disk.

    Samples are only kept when exactly one face is visible — two people in
    frame during enrolment is the fastest way to poison a model.
    """
    config.ensure_dirs()
    target = config.faces_dir / str(label)
    target.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(config.face_size)
    saved = 0

    with open_camera(config.camera_index, config.warmup_frames) as capture:
        while saved < config.samples_per_student:
            frame = read_frame(capture)
            boxes = detector.detect(frame)

            if len(boxes) == 1:
                face = detector.crop(frame, boxes[0])
                cv2.imwrite(str(target / f"{saved:03d}.png"), face)
                saved += 1
                if on_progress:
                    on_progress(saved, config.samples_per_student)

            if show_preview:
                _draw_boxes(frame, boxes)
                cv2.putText(
                    frame,
                    f"{saved}/{config.samples_per_student}  (q to abort)",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Enrolment", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show_preview:
        cv2.destroyAllWindows()
    return saved


def load_dataset(config: Config) -> tuple[list[np.ndarray], list[int]]:
    """Read every stored sample back off disk, keyed by label."""
    faces: list[np.ndarray] = []
    labels: list[int] = []
    if not config.faces_dir.exists():
        return faces, labels

    for label_dir in sorted(config.faces_dir.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        for image_path in sorted(label_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if image.shape != tuple(reversed(config.face_size)):
                image = cv2.resize(image, config.face_size, interpolation=cv2.INTER_AREA)
            faces.append(image)
            labels.append(label)
    return faces, labels


def train(config: Config) -> tuple[int, int]:
    """Train the recogniser from stored samples. Returns (samples, identities)."""
    faces, labels = load_dataset(config)
    recognizer = FaceRecognizer(config.max_distance)
    recognizer.train(faces, labels)
    recognizer.save(config.model_path)
    return len(faces), len(set(labels))


def remove_samples(config: Config, label: int) -> None:
    """Delete a student's stored samples, e.g. after de-registration."""
    target = config.faces_dir / str(label)
    if target.exists():
        for path in target.glob("*.png"):
            path.unlink()
        target.rmdir()


def _draw_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
