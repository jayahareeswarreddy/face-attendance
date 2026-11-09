"""Marking attendance from a live camera."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import cv2

from .camera import open_camera, read_frame
from .config import Config
from .detector import FaceDetector
from .recognizer import FaceRecognizer
from . import storage


@dataclass(frozen=True)
class Outcome:
    """What happened when we pointed the camera at someone."""

    status: str  # recognised | already_marked | unknown_face | no_face | unknown_label
    student: storage.Student | None = None
    distance: float | None = None

    @property
    def message(self) -> str:
        if self.status == "recognised" and self.student:
            return (
                f"Attendance marked for {self.student.name} "
                f"({self.student.student_id}) — distance {self.distance:.1f}"
            )
        if self.status == "already_marked" and self.student:
            return f"{self.student.name} is already marked present today"
        if self.status == "unknown_face":
            return (
                f"Face not recognised (distance {self.distance:.1f}). "
                "Enrol first, or retrain if lighting has changed."
            )
        if self.status == "unknown_label":
            return "Model returned a label with no matching student — retrain"
        return "No face detected"


def mark_once(
    config: Config,
    conn: sqlite3.Connection,
    attempts: int = 40,
    show_preview: bool = True,
) -> Outcome:
    """Watch the camera until a face is recognised or attempts run out.

    The identity is whatever the model says. Nothing the person at the
    keyboard types can influence which record gets written — that is the
    entire point of the rewrite.
    """
    detector = FaceDetector(config.face_size)
    recognizer = FaceRecognizer(config.max_distance)
    recognizer.load(config.model_path)

    outcome = Outcome("no_face")

    with open_camera(config.camera_index, config.warmup_frames) as capture:
        for _ in range(attempts):
            frame = read_frame(capture)
            box = detector.largest(frame)

            if box is not None:
                match = recognizer.predict(detector.crop(frame, box))
                if match.accepted:
                    student = storage.get_by_label(conn, match.label)
                    if student is None:
                        outcome = Outcome("unknown_label", None, match.distance)
                    else:
                        created = storage.mark_attendance(
                            conn, student.student_id, match.distance
                        )
                        outcome = Outcome(
                            "recognised" if created else "already_marked",
                            student,
                            match.distance,
                        )
                    break
                outcome = Outcome("unknown_face", None, match.distance)

            if show_preview:
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(
                    frame, "Looking... (q to cancel)", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
                )
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show_preview:
        cv2.destroyAllWindows()
    return outcome
