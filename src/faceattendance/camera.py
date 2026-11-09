"""Camera handling.

Wrapped in a context manager because the original prototype leaked the
capture device on every early return.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np


class CameraError(RuntimeError):
    """The camera could not be opened or produced no usable frame."""


@contextmanager
def open_camera(index: int = 0, warmup_frames: int = 12) -> Iterator["cv2.VideoCapture"]:
    """Yield an opened camera, always releasing it on the way out.

    Most webcams return a dark or garbage frame for the first few reads
    while exposure settles, so we discard a short warm-up burst before
    handing the device over.
    """
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(f"could not open camera at index {index}")
    try:
        for _ in range(max(0, warmup_frames)):
            capture.read()
        yield capture
    finally:
        capture.release()


def read_frame(capture: "cv2.VideoCapture") -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise CameraError("camera returned no frame")
    return frame
