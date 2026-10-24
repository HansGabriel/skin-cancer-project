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

## 2. The overnight retrain (started 24 Oct on the PC; 3–5 h)

Log: `training/runs/2026-10-24-retrain.log`, finished notebook
`training/runs/2026-10-24-retrain.ipynb`. The notebook wrote **straight into `models/`**
(all four files) and `training/splits.csv`; nothing is committed. The previous model is
the committed one (`git show main:models/skin_classifier.tflite`) and a copy sits in
`datasets/models_oct18_backup/`.

Read the last `external_eval.py` table in the log (rows per source, column `flagged`).
**Ship only if all three hold:**

1. `MILK10k · clinical: close-up` benign **spec** rose from 0.37 toward 0.7
2. `MILK10k` **flagged ≥ 28** of 31 cancers
3. `ALL` **flagged ≥ 121** of 149 cancers

- All three hold → `git add models/ training/splits.csv`, paste the new tables into
  `docs/METRICS.md` under a new dated heading (never mix two models' numbers), commit,
  push, let Hans review. Railway redeploys from `main`; the Pi updates with the button.
- Any fails → `git checkout -- models/ training/splits.csv` and paste the numbers into
  `docs/METRICS.md` anyway, under open item 4, as "tried, not shipped".

Never judge a model by the drawer's bars: the decision is `cancer score >= limit`, printed
in the DECISION line.

## 3. Set the gate limits from real photos (30 min with the cone)

Steps are in `docs/DEPLOYMENT.md`, "Setting the gate limits from real photos". Faces never
go into git; `captures/` is ignored.

## Parked

`no_lesion` class (normal-skin photos need SRC/IRB approval, `docs/PRIVACY.md`); batching
the four TTA views on the Pi; the `spot_at_edge` rule for centred spots that touch the
border (`docs/METRICS.md` item 5).

The rules that bite are in `AGENTS.md`; `pytest -q` must be green before any push.
