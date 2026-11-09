"""Tests for the data layer. No camera and no OpenCV required."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from faceattendance import storage


@pytest.fixture()
def conn(tmp_path: Path):
    with storage.connect(tmp_path / "test.db") as connection:
        yield connection


def test_labels_are_allocated_sequentially(conn):
    a = storage.add_student(conn, "S1", "Asha")
    b = storage.add_student(conn, "S2", "Bala")
    assert (a.label, b.label) == (0, 1)


def test_duplicate_student_id_is_rejected(conn):
    storage.add_student(conn, "S1", "Asha")
    with pytest.raises(storage.StudentExists):
        storage.add_student(conn, "S1", "Someone Else")


def test_blank_fields_are_rejected(conn):
    with pytest.raises(ValueError):
        storage.add_student(conn, "   ", "Asha")


def test_attendance_is_once_per_day(conn):
    storage.add_student(conn, "S1", "Asha")
    first = storage.mark_attendance(conn, "S1", 31.0)
    second = storage.mark_attendance(conn, "S1", 29.5)
    assert first is True
    assert second is False
    assert len(storage.list_attendance(conn)) == 1


def test_attendance_on_a_new_day_is_allowed(conn):
    storage.add_student(conn, "S1", "Asha")
    today = datetime(2026, 3, 1, 9, 0)
    storage.mark_attendance(conn, "S1", 30.0, when=today)
    storage.mark_attendance(conn, "S1", 30.0, when=today + timedelta(days=1))
    assert len(storage.list_attendance(conn)) == 2


def test_marking_an_unknown_student_raises(conn):
    with pytest.raises(storage.UnknownStudent):
        storage.mark_attendance(conn, "ghost", 10.0)


def test_deleting_a_student_cascades_to_attendance(conn):
    storage.add_student(conn, "S1", "Asha")
    storage.mark_attendance(conn, "S1", 30.0)
    storage.delete_student(conn, "S1")
    assert storage.list_students(conn) == []
    assert storage.list_attendance(conn) == []


def test_lookup_by_label_round_trips(conn):
    created = storage.add_student(conn, "S1", "Asha")
    assert storage.get_by_label(conn, created.label).student_id == "S1"


def test_export_csv_writes_a_header_and_every_row(conn, tmp_path: Path):
    storage.add_student(conn, "S1", "Asha")
    storage.mark_attendance(conn, "S1", 30.0)
    destination = tmp_path / "out" / "attendance.csv"

    count = storage.export_csv(conn, destination)

    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    assert count == 1
    assert lines[0] == "student_id,name,day,marked_at,distance"
    assert lines[1].startswith("S1,Asha,")
