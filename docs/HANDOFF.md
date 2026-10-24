# Handoff to the team — 24 October 2026

Read `AGENTS.md` first. This page is the short list of what is running, what is still
owed, and how to tell whether the overnight retrain may be shipped. Hans pushes to
`main` on GitHub; the Pi pulls with the **Update from GitHub** button (`docs/DEPLOYMENT.md`).

## What is on `main` now

- Photo gate refuses a face, a palm or a desk before the model runs (`not_in_ring`,
  `several_marks`, `fills_frame`), and the check screen says so in one line:
  **THIS DOES NOT LOOK LIKE ONE SPOT ON SKIN** (`dermascan/verdict.py`).
- **Update from GitHub** in Settings: fetch → fast-forward `main` → `pip install` only if
  `requirements.txt` changed → `pytest` → restart; a failure rolls back. Needs wifi.
- Gate cost measured (`docs/METRICS.md`, open item 5): the limits stay 0.40 / 0.25 and must
  be set from cone captures, not from the bench.

## 1. Update the Pi (5 min, first thing)

```bash
cd ~/Documents/skin-cancer-project && git branch --show-current   # must say main
git pull && source venv/bin/activate && pytest -q                 # once; after this, use the button
```

Restart the kiosk (double-click EPIVUE). Settings now shows **Software** and **Last update**.
Then take a forced scan ("Check it anyway") and open **Details for staff**: if the word
`null` appears where the A B C rows go, send Hans a photo of the screen (not reproducible
on `main`; it may have been the old code).

## 2. The 24 October retrain — done, not shipped

Result is in `docs/METRICS.md` ("24 October 2026: … NOT shipped"): MILK close-up
specificity 0.37 → 0.525 and 28/31 MILK cancers held, but total cancers flagged fell to
113/149 (rule: ≥ 121). `models/` is still the 18 October model. The tried model and its
log are in `datasets/models_2026-10-24_stratified/` on the PC (not in git).

To try the next idea (a per-image-type threshold in `training/calibrate.py`), run the
notebook again from `jupyter lab training/train_skin_classifier.ipynb` (cell 0 sets the
GPU) or headless:

```bash
VIRTUAL_ENV=$PWD/venv PATH=$PWD/venv/bin:$PATH venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 training/train_skin_classifier.ipynb \
  --output-dir training/runs --output $(date +%F)-retrain.ipynb
```

It writes **straight into `models/`**. Read the last cell's `external_eval.py` table and
apply the ship rule — all three must hold:

1. `MILK10k · clinical: close-up` benign **spec** rose from 0.37 toward 0.7
2. `MILK10k` **flagged ≥ 28** of 31 cancers
3. `ALL` **flagged ≥ 121** of 149 cancers

Pass → `git add models/ training/splits.csv`, paste the tables into `docs/METRICS.md`
under a new dated heading (never mix two models' numbers), push a branch for Hans.
Fail → `git checkout -- models/ training/splits.csv` and record the numbers anyway.
Never judge a model by the drawer's bars: the decision is `cancer score >= limit`.

## 3. Set the gate limits from real photos (30 min with the cone)

Steps are in `docs/DEPLOYMENT.md`, "Setting the gate limits from real photos". Faces never
go into git; `captures/` is ignored.

## Parked

`no_lesion` class (normal-skin photos need SRC/IRB approval, `docs/PRIVACY.md`); batching
the four TTA views on the Pi; the `spot_at_edge` rule for centred spots that touch the
border (`docs/METRICS.md` item 5).

The rules that bite are in `AGENTS.md`; `pytest -q` must be green before any push.
