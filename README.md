# Face Attendance

Attendance marked by recognising a face, not by trusting what someone typed.

## The problem this fixes

The obvious way to build this is the wrong way. You ask the student for their ID,
point a camera at them, check that a face is visible, and write the record. It looks
like a face attendance system and it demos perfectly.

It is also trivially defeated. A Haar cascade answers *"is there a face here?"* — not
*"whose face is this?"* So any student can type a friend's ID, show their own face, and
mark that friend present. The camera contributes nothing but theatre.

This version inverts the flow. There is no field to type an ID into when marking
attendance. The recogniser produces a label, the label resolves to a student, and that
is the record that gets written. **A user cannot assert an identity — only a face can.**

## How it works

```
enrol ──► capture 25 normalised face crops ──► data/faces/<label>/
                                                      │
train ──────────────────────────────────────────► LBPH model.yml
                                                      │
mark  ──► detect largest face ──► predict ──► distance ≤ threshold?
                                                 │              │
                                                yes             no
                                                 │              │
                                        write attendance   "not recognised"
```

**Enrolment** captures samples only while exactly one face is in frame — two people
visible during enrolment is the fastest way to poison a model. Each crop is converted
to grayscale, resized to 200×200 and histogram-equalised, because LBPH is sensitive to
lighting and nobody enrols under the same lamp they later attend under.

**Recognition** uses LBPH from `opencv-contrib`. It returns a *distance*, where lower
means a closer match; anything above the configured threshold is reported as an
unknown face rather than a weak guess. Training refuses to run with fewer than two
enrolled students, because a single-identity model matches everybody who sits in front
of it — a failure mode that looks like success.

**Storage** is SQLite with a `UNIQUE (student_id, day)` constraint, so the
once-per-day rule is enforced by the database rather than by a read-then-write check
that two callers could race through.

## Layout

```
src/faceattendance/
  config.py       paths and thresholds, all tunables in one frozen dataclass
  storage.py      SQLite schema and queries — imports no OpenCV, so it is testable
  camera.py       context-managed capture with an exposure warm-up
  detector.py     Haar detection, cropping and normalisation
  recognizer.py   LBPH train/predict/save/load behind a swappable interface
  enrollment.py   sample capture and model training
  attendance.py   the marking session
  cli.py          argparse entry point
tests/            13 tests, no camera required
```

The recogniser is isolated behind one class. Replacing LBPH with an embedding model
means rewriting `recognizer.py` and nothing else.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -e ".[dev]"
```

`opencv-contrib-python` is required, not `opencv-python` — `cv2.face` lives only in the
contrib build. The CLI says so plainly if you install the wrong one.

## Use

```bash
faceattendance enrol S101 "Asha Rao"    # register, then look at the camera
faceattendance enrol S102 "Bala Kumar"  # at least two students before training
faceattendance train                    # build the model
faceattendance mark                     # recognise whoever is in frame

faceattendance students                 # who is registered
faceattendance records --today          # today's attendance
faceattendance export attendance.csv    # dump everything to CSV
faceattendance remove S101              # delete a student, then retrain
```

Useful flags: `--data-dir` to relocate the database, samples and model;
`--camera N` to pick a device; `--no-preview` to run without an OpenCV window.

## Tests

```bash
pytest
```

The data layer and the CLI parser are covered without a camera or a trained model —
the once-per-day guarantee, cascade deletes, label allocation and CSV export all have
tests. The vision path is left to manual verification, since it needs a face.

## Limitations

Worth stating plainly, because a system like this invites more trust than it has
earned:

- **LBPH is not strong authentication.** It is well suited to a classroom of known
  students and is not hard to fool with a photograph. There is no liveness detection.
- Accuracy degrades when lighting differs sharply from enrolment conditions. Retrain,
  or enrol under varied lighting.
- Only the largest face in frame is considered, so it marks one person at a time.
- Face data is personal data. Samples and the model stay on the local disk and `data/`
  is gitignored, but deployment anywhere real means asking for consent first.

## History

This repository began as a small classroom prototype that detected a face while the
student typed their own ID. This is a ground-up rewrite with that flaw as its starting
point; the original script is preserved in the first commit.

## License

MIT — see [LICENSE](LICENSE).
