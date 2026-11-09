"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from .config import Config
from . import storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceattendance",
        description="Attendance by face recognition.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="where the database, samples and model live (default: ./data)",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument(
        "--no-preview", action="store_true",
        help="run headless, without an OpenCV preview window",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enrol = sub.add_parser("enrol", help="register a student and capture their face")
    enrol.add_argument("student_id")
    enrol.add_argument("name")
    enrol.add_argument("--samples", type=int, default=25)

    sub.add_parser("train", help="train the recogniser from captured samples")
    sub.add_parser("mark", help="mark attendance for whoever is in front of the camera")
    sub.add_parser("students", help="list registered students")

    records = sub.add_parser("records", help="show attendance records")
    records.add_argument("--today", action="store_true", help="only today's records")

    export = sub.add_parser("export", help="write all records to CSV")
    export.add_argument("destination", type=Path)

    remove = sub.add_parser("remove", help="delete a student and their data")
    remove.add_argument("student_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.load(
        args.data_dir,
        camera_index=args.camera,
        samples_per_student=getattr(args, "samples", 25),
    )
    config.ensure_dirs()
    preview = not args.no_preview

    with storage.connect(config.db_path) as conn:
        return _dispatch(args, config, conn, preview)


def _dispatch(args, config: Config, conn, preview: bool) -> int:
    command = args.command

    if command == "enrol":
        return _enrol(args, config, conn, preview)

    if command == "train":
        from .enrollment import train

        try:
            samples, identities = train(config)
        except (ValueError, RuntimeError) as exc:
            print(f"Cannot train: {exc}", file=sys.stderr)
            return 1
        print(f"Trained on {samples} samples across {identities} students.")
        return 0

    if command == "mark":
        from .attendance import mark_once
        from .recognizer import ModelNotTrained

        try:
            outcome = mark_once(config, conn, show_preview=preview)
        except (ModelNotTrained, RuntimeError) as exc:
            print(f"Cannot mark attendance: {exc}", file=sys.stderr)
            return 1
        print(outcome.message)
        return 0 if outcome.status in {"recognised", "already_marked"} else 1

    if command == "students":
        students = storage.list_students(conn)
        if not students:
            print("No students registered.")
            return 0
        print(f"{'ID':<14}{'NAME':<28}{'LABEL':<7}ENROLLED")
        for s in students:
            print(f"{s.student_id:<14}{s.name:<28}{s.label:<7}{s.enrolled_at}")
        return 0

    if command == "records":
        day = date.today() if args.today else None
        records = storage.list_attendance(conn, day)
        if not records:
            print("No attendance records found.")
            return 0
        print(f"{'ID':<14}{'NAME':<28}{'MARKED AT':<22}DIST")
        for r in records:
            print(f"{r.student_id:<14}{r.name:<28}{r.marked_at:<22}{r.distance:.1f}")
        return 0

    if command == "export":
        count = storage.export_csv(conn, args.destination)
        print(f"Wrote {count} records to {args.destination}")
        return 0

    if command == "remove":
        return _remove(args, config, conn)

    return 1


def _enrol(args, config: Config, conn, preview: bool) -> int:
    from .enrollment import capture_samples

    try:
        student = storage.add_student(conn, args.student_id, args.name)
    except storage.StudentExists:
        print(f"{args.student_id} is already registered.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Registered {student.name} as label {student.label}. Look at the camera.")

    def progress(done: int, total: int) -> None:
        print(f"\r  captured {done}/{total}", end="", flush=True)

    saved = capture_samples(config, student.label, progress, show_preview=preview)
    print()

    if saved == 0:
        storage.delete_student(conn, student.student_id)
        print("No face captured — registration rolled back.", file=sys.stderr)
        return 1

    print(f"Saved {saved} samples. Run 'faceattendance train' to rebuild the model.")
    return 0


def _remove(args, config: Config, conn) -> int:
    from .enrollment import remove_samples

    student = storage.get_student(conn, args.student_id)
    if student is None:
        print(f"No student with id {args.student_id}.", file=sys.stderr)
        return 1
    storage.delete_student(conn, student.student_id)
    remove_samples(config, student.label)
    print(f"Removed {student.name}. Retrain so the model forgets them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
