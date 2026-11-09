"""Tests for argument parsing, which needs neither a camera nor a model."""

from __future__ import annotations

import pytest

from faceattendance.cli import build_parser


def test_enrol_requires_id_and_name():
    args = build_parser().parse_args(["enrol", "S1", "Asha"])
    assert (args.command, args.student_id, args.name) == ("enrol", "S1", "Asha")


def test_a_subcommand_is_mandatory():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_records_defaults_to_all_days():
    assert build_parser().parse_args(["records"]).today is False


def test_preview_is_on_unless_disabled():
    assert build_parser().parse_args(["mark"]).no_preview is False
    assert build_parser().parse_args(["--no-preview", "mark"]).no_preview is True
