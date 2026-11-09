"""Face detection and normalisation."""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]


class FaceDetector:
    """Haar cascade detector returning normalised grayscale face crops."""

    def __init__(self, face_size: tuple[int, int] = (200, 200)) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {cascade_path}")
        self._face_size = face_size

    def detect(self, frame: np.ndarray) -> list[Box]:
        """Return every face bounding box found in a BGR frame."""
        gray = self._to_gray(frame)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def largest(self, frame: np.ndarray) -> Box | None:
        """Return the biggest face in the frame, which is the closest one."""
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda box: box[2] * box[3])

    def crop(self, frame: np.ndarray, box: Box) -> np.ndarray:
        """Extract a face and normalise it for the recogniser.

        Histogram equalisation matters here: LBPH is sensitive to lighting,
        and enrolment rarely happens under the same lamp as attendance.
        """
        x, y, w, h = box
        gray = self._to_gray(frame)
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, self._face_size, interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(face)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
