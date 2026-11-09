"""SQLite persistence for students and attendance records.

Deliberately free of any OpenCV import so the data layer can be tested
without a camera or the contrib build installed.
"""

from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    label       INTEGER NOT NULL UNIQUE,
    enrolled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    day        TEXT NOT NULL,
    marked_at  TEXT NOT NULL,
    distance   REAL NOT NULL,
    UNIQUE (student_id, day)
);

CREATE INDEX IF NOT EXISTS idx_attendance_day ON attendance(day);
"""


@dataclass(frozen=True)
class Student:
    student_id: str
    name: str
    label: int
    enrolled_at: str


@dataclass(frozen=True)
class Record:
    student_id: str
    name: str
    day: str
    marked_at: str
    distance: float


class StudentExists(Exception):
    """Raised when enrolling a student_id that is already registered."""


class UnknownStudent(Exception):
    """Raised when an operation targets a student_id that does not exist."""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database with foreign keys on, committing on clean exit."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def add_student(conn: sqlite3.Connection, student_id: str, name: str) -> Student:
    """Register a student and allocate the next recogniser label."""
    student_id, name = student_id.strip(), name.strip()
    if not student_id or not name:
        raise ValueError("student_id and name must both be non-empty")

    if conn.execute(
        "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
    ).fetchone():
        raise StudentExists(student_id)

    row = conn.execute("SELECT COALESCE(MAX(label), -1) + 1 AS next FROM students").fetchone()
    label = int(row["next"])
    enrolled_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO students (student_id, name, label, enrolled_at) VALUES (?, ?, ?, ?)",
        (student_id, name, label, enrolled_at),
    )
    return Student(student_id, name, label, enrolled_at)


def get_student(conn: sqlite3.Connection, student_id: str) -> Student | None:
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    return _to_student(row) if row else None


def get_by_label(conn: sqlite3.Connection, label: int) -> Student | None:
    row = conn.execute("SELECT * FROM students WHERE label = ?", (label,)).fetchone()
    return _to_student(row) if row else None


def list_students(conn: sqlite3.Connection) -> list[Student]:
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    return [_to_student(r) for r in rows]


def delete_student(conn: sqlite3.Connection, student_id: str) -> None:
    """Remove a student and, by cascade, their attendance history."""
    cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    if cur.rowcount == 0:
        raise UnknownStudent(student_id)


def mark_attendance(
    conn: sqlite3.Connection,
    student_id: str,
    distance: float,
    when: datetime | None = None,
) -> bool:
    """Record attendance for a day.

    Returns True if a new record was written, False if the student was
    already marked present that day. The uniqueness guarantee is the
    database's, not a read-then-write check, so concurrent callers cannot
    both succeed.
    """
    if get_student(conn, student_id) is None:
        raise UnknownStudent(student_id)

    when = when or datetime.now()
    cur = conn.execute(
        "INSERT OR IGNORE INTO attendance (student_id, day, marked_at, distance) "
        "VALUES (?, ?, ?, ?)",
        (student_id, when.date().isoformat(), when.isoformat(timespec="seconds"), distance),
    )
    return cur.rowcount == 1


def list_attendance(
    conn: sqlite3.Connection, day: date | None = None
) -> list[Record]:
    sql = (
        "SELECT a.student_id, s.name, a.day, a.marked_at, a.distance "
        "FROM attendance a JOIN students s USING (student_id) "
    )
    params: tuple[object, ...] = ()
    if day is not None:
        sql += "WHERE a.day = ? "
        params = (day.isoformat(),)
    sql += "ORDER BY a.marked_at DESC"
    return [
        Record(r["student_id"], r["name"], r["day"], r["marked_at"], r["distance"])
        for r in conn.execute(sql, params).fetchall()
    ]


def export_csv(conn: sqlite3.Connection, destination: Path) -> int:
    """Write every attendance record to CSV. Returns the row count."""
    records = list_attendance(conn)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["student_id", "name", "day", "marked_at", "distance"])
        for r in records:
            writer.writerow([r.student_id, r.name, r.day, r.marked_at, f"{r.distance:.2f}"])
    return len(records)


def _to_student(row: sqlite3.Row) -> Student:
    return Student(row["student_id"], row["name"], row["label"], row["enrolled_at"])
