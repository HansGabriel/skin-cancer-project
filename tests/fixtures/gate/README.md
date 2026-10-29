# Real photos for the gate

**Both folders are empty as of 29 October 2026** (only `.gitkeep`), so
`tests/test_gate_real.py` runs no cases and every gate limit is still a guess
(`dermascan/config.py`, the lines marked `GUESSED`). The 30 October Pi capture session
is meant to fill them: a few dozen refuse/pass photos from the cone, per the rules below.

Small real JPEGs the photo gate (`dermascan/gate.py`) must get right. The synthetic
frames in `tests/conftest.py` catch regressions in the maths; these catch the cases
that only real cameras produce. `tests/test_gate_real.py` runs every file here
through the real decode path (`dermascan.scan.check_photo`).

| folder    | the gate must         | what goes here                                                   |
|-----------|-----------------------|------------------------------------------------------------------|
| `refuse/` | refuse (any code)     | palms, whole hands, rooms, desks, walls, beige surfaces, clothes |
| `pass/`   | pass                  | one lesion close-up per file: Pi cone captures (with consent) or ISIC images |

Rules:

- **No faces of real people in the repository.** `main` auto-deploys to Railway. Keep
  face captures in the gitignored `captures/refuse/` folder at the repo root; the
  test scans `captures/refuse/` and `captures/pass/` too when they exist.
- ≤ 512 px on the long edge, JPEG quality 85, under ~80 KB each. The gate works on a
  384 px copy and focus on a 512 px copy (`config.GATE_WORK_PX`, `FOCUS_WORK_PX`), so a
  512 px file gives the gate the same pixels a full capture would.
  Mac: `sips -Z 512 -s format jpeg -s formatOptions 85 in.jpg --out tests/fixtures/gate/refuse/palm-webcam.jpg`
- Name files by what they show: `palm-webcam.jpg`, `hand-phone.jpg`, `desk-pi.jpg`,
  `nevus-pi-cone.jpg`.
- ISIC images are CC-BY-NC: add a row to `pass/attribution.csv`
  (`file,isic_id,attribution,license,url`) copied from `training/external_bench.csv`.
- Pi captures of a person's skin need that person's consent; note it in the commit.
  `docs/PRIVACY.md` is the promise the kiosk makes ("nothing leaves the device"): a
  photo only comes here if its owner agreed to that one exception.

To see the numbers behind a pass or refusal, or to pick a limit from a folder of photos:

    python scripts/gate_report.py tests/fixtures/gate/refuse --expect refuse
    python scripts/gate_report.py tests/fixtures/gate/pass --expect pass --full
