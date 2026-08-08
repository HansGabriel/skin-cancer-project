# Final Improvements — Primary-Source Research

**Date:** 2026-08-08. **Purpose:** primary-source verification for the last round of DermaScan improvements: (A) leakage-free HAM10000 retrain split, (B) explainability on the Pi without TensorFlow, (C) kiosk hardening on current Raspberry Pi OS, (D) human-participants rules before photographing students' skin. Every claim is followed by its source (official docs, source code, standards bodies, or original papers). Gaps are listed at the end, not papered over.

---

## Track A — Leakage-free HAM10000 splitting

**VERDICT:** Confirmed. HAM10000's 10,015 images cover fewer unique lesions (multiple photos per lesion, linked by `lesion_id`), so the current image-level `train_test_split` leaks lesions across train/test — a failure mode documented for ISIC datasets in a peer-reviewed source. Split by `lesion_id` (pattern below); report metrics only from the lesion-grouped test set.

- The dataset deliberately contains multiple images of the same lesion: "The number of images in the datasets does not correspond to the number of unique lesions, because we also provide images of the same lesion taken at different magnifications or angles, or with different cameras."
  — Tschandl, Rosendahl & Kittler (2018), *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Scientific Data 5:180161 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/ (open-access mirror of https://doi.org/10.1038/sdata.2018.161). Classes: akiec, bcc, bkl, df, mel, nv, vasc.
- The official Harvard Dataverse description states lesions with multiple images are tracked via the `lesion_id` column of `HAM10000_metadata`; the deposit also ships the official ISIC2018 Task 3 test set (`ISIC2018_Task3_Test_Images.zip`, 1,511 images + ground truth) usable as an untouched external test set.
  — Harvard Dataverse, doi:10.7910/DVN/DBW86T — https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T (verified via the JSON metadata export API).
- Citable leakage evidence: "Our analysis found a significant number of duplicate images, both within and between the datasets" and "we also noted duplicates spread across testing and training sets"; the authors removed 14,310 duplicates from the training set to build a clean benchmark.
  — Cassidy, Kendrick, Brodzicki, Jaworek-Korjakowska & Yap (2022), *Analysis of the ISIC image datasets: Usage, benchmarks and recommendations*, Medical Image Analysis 75:102305, doi:10.1016/j.media.2021.102305 — abstract verified at https://pubmed.ncbi.nlm.nih.gov/34852988/.
- scikit-learn 1.8 API (versioned docs):
  - `StratifiedGroupKFold` — "Class-wise stratified K-Fold iterator variant with non-overlapping groups"; it only "attempts to return stratified folds" (best-effort), and "when there is a small number of groups containing a large number of samples the stratification will not be possible and the behavior will be close to GroupKFold". Params: `n_splits=5`, `shuffle=False`, `random_state=None`.
    — https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
  - `GroupShuffleSplit` — whole groups go to train or test; "The parameters test_size and train_size refer to groups, and not to samples as in ShuffleSplit"; it does **not** stratify by class.
    — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
  - `train_test_split(stratify=...)` — "If not None, data is split in a stratified fashion, using this as the class labels."
    — https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.train_test_split.html

**Recommended 70/15/15 lesion-grouped stratified split** (exact stratification at lesion level; group safety by construction — dx is constant within a lesion, asserted):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

meta = pd.read_csv("HAM10000_metadata.csv")            # image_id, lesion_id, dx
per_lesion = meta.groupby("lesion_id")["dx"].agg(["first", "nunique"])
assert (per_lesion["nunique"] == 1).all()              # dx constant per lesion
lids, ldx = per_lesion.index.to_numpy(), per_lesion["first"].to_numpy()

train_l, tmp_l, _, tmp_y = train_test_split(
    lids, ldx, test_size=0.30, stratify=ldx, random_state=42)
val_l, test_l, _, _ = train_test_split(
    tmp_l, tmp_y, test_size=0.50, stratify=tmp_y, random_state=42)

lookup = ({l: "train" for l in train_l} | {l: "val" for l in val_l}
          | {l: "test" for l in test_l})
meta["split"] = meta["lesion_id"].map(lookup)
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not (set(meta.loc[meta.split == a, "lesion_id"])
                & set(meta.loc[meta.split == b, "lesion_id"]))
```

Caveat: ratios/stratification are exact in lesions, approximate in images (lesions have 1..n images) — the same trade-off sklearn documents for `StratifiedGroupKFold`. Use `StratifiedGroupKFold(groups=lesion_id)` for any k-fold CV experiments. This works unchanged for a binary benign/suspicious remap (stratify on the remapped label).

---

## Track B — Explainability on the Pi without TensorFlow

**VERDICT:** Feasible under ai-edge-litert. Best path: re-export the retrained model with two outputs (softmax + last conv feature map) and run a gradient-free CAM — Eigen-CAM (1 forward pass) or Score-CAM (~1 forward pass per channel, expensive). Grad-CAM is ruled out (needs gradients; litert is inference-only). Classic CAM only becomes available if the retrain flattens the head to GAP→Dense(3).

1. **Multiple outputs + intermediate tensors (LiteRT Python API).** `get_output_details()` "Gets model output tensor details. Returns: A list in which each item is a dictionary with details about an output tensor" — i.e., multi-output models are first-class. `get_tensor()`/`tensor()` state "This function cannot be used to read intermediate results."; reading intermediates requires the constructor flag `experimental_preserve_all_tensors`: "If true, then intermediate tensors used during computation are preserved for inspection … If false, getting intermediate tensors could result in undefined values or None, especially when the graph is successfully modified by the Tensorflow Lite default delegate."
   — TFLite Python Interpreter source (docstrings) — https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/python/interpreter.py. The same file carries: "tf.lite.Interpreter is deprecated … Please use the LiteRT interpreter from the ai_edge_litert package."
   The LiteRT migration guide confirms drop-in parity: "LiteRT fully supports the TensorFlow Lite Interpreter API, migrating requires only a package name update—no logic changes are necessary" (`pip install ai-edge-litert`; `from ai_edge_litert.interpreter import Interpreter`). — https://developers.google.com/edge/litert/migration (redirect target of ai.google.dev/edge/litert/migration). Preferring a declared second output over `experimental_preserve_all_tensors` avoids the delegate/undefined-value caveat and INT8 dequant guesswork.
2. **Converting a multi-output Keras model.** `tf.lite.TFLiteConverter.from_keras_model(model)` (or `from_saved_model`, recommended) is the documented path — https://developers.google.com/edge/litert/models/convert_tf. Signatures preserve named outputs: outputs are a "Map for output mapping from output name in signature to an output tensor"; from Python, `interpreter.get_signature_runner()` returns "dictionaries with all outputs from the inference"; "Keras model converter API uses the default signature automatically." — https://developers.google.com/edge/litert/models/signatures. Use the signature runner so outputs are fetched by name, not by index order.
3. **Gradient-free CAM methods (original papers).**
   - Score-CAM: "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks" (Wang et al., CVPR 2020 Workshops) — https://arxiv.org/abs/1910.01279 — removes gradient dependence; each activation map's weight comes from "its forward passing score on target class", final map = "a linear combination of weights and activation maps". Cost: one masked forward pass per channel of the target layer — for EfficientNetB0's 1280-channel top conv that is ~1280 inferences per explanation on the Pi (repo-derived sizing, not from the paper; a top-k channel subset is the usual mitigation).
   - Eigen-CAM: "Eigen-CAM: Class Activation Map using Principal Components" (Muhammad & Yeasin, IJCNN 2020, doi:10.1109/IJCNN48605.2020.9206626) — https://arxiv.org/abs/2008.00299 — "does not rely on the backpropagation of gradients, class relevance score, maximum activation locations, or any other form of weighting features": a single forward pass plus an SVD of the conv activations. Caveat: class-agnostic (one map regardless of predicted class).
   - Classic CAM: *Learning Deep Features for Discriminative Localization* (Zhou et al., CVPR 2016) — https://arxiv.org/abs/1512.04150 — requires global average pooling feeding the final classification layer (CAM = output-layer weights projected onto conv maps). The current head (GAP → Dropout → Dense(128) → Dropout → Dense(3)) breaks that assumption; CAM applies only if the retrain flattens the head to GAP→Dense(3), in which case CAM is essentially free at inference.
   - Grad-CAM: (Selvaraju et al., ICCV 2017) — https://arxiv.org/abs/1610.02391 — "uses the gradients of any target concept, flowing into the final convolutional layer". The litert Interpreter exposes no gradient/autodiff API (inference only, per the Interpreter API above) → Grad-CAM cannot run under ai-edge-litert.
4. **Fallback (full TF on the Pi).** Official aarch64 Linux wheels exist: "Starting with TensorFlow 2.10, Linux CPU-builds for Aarch64/ARM64 processors are built, maintained, tested and released by a third party: AWS." — https://www.tensorflow.org/install/pip. PyPI ships e.g. `tensorflow-2.21.0-cp312-cp312-manylinux_2_27_aarch64.whl` (~282 MB wheel; CPython 3.10–3.13, glibc 2.27+ — satisfied by Pi OS Bookworm/Trixie) — https://pypi.org/project/tensorflow/#files. RAM: no authoritative TF-on-Pi memory figure exists (see Open questions); the Pi 4 ships with "LPDDR4-3200 SDRAM" in 2GB/4GB/8GB variants — https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/ — and LiteRT is Google's runtime for "on-device machine learning inference with low latency and a small binary size" — https://pypi.org/project/ai-edge-litert/. Recommendation: avoid full TF on the 2GB unit; stay on litert. ai-edge-litert itself publishes aarch64 wheels (2.1.6 current, 2026-07-01; the pinned 2.1.5 exists, 2026-05-15; e.g. `ai_edge_litert-2.1.6-cp314-cp314-manylinux_2_27_aarch64.whl`) — https://pypi.org/project/ai-edge-litert/#files.

---

## Track C — Kiosk hardening on current Raspberry Pi OS

**VERDICT:** docs/DEPLOYMENT.md never names the OS release, but its `/boot/firmware/config.txt` path implies Bookworm-or-later — where the desktop is Wayland/labwc, the old LXDE autostart is dead, and surf only runs via XWayland. Recommended: labwc `autostart` (official kiosk method) plus a `systemd --user` unit with `Restart=on-failure` bound to `graphical-session.target`, autologin enabled; both X11 and Wayland paths below.

- **Current OS & compositor.** "Raspberry Pi Desktop now runs Wayland by default across all models" with labwc replacing Wayfire (release of 2024-10-28); X11 apps keep working because "labwc includes a library called Xwayland, which provides a virtual X implementation running on top of Wayland. labwc provides this virtual implementation automatically for any application that isn't compatible with Wayland."
  — Raspberry Pi (official news), *A new release of Raspberry Pi OS* — https://www.raspberrypi.com/news/a-new-release-of-raspberry-pi-os/. The current release is Trixie (Debian 13, Oct 2025), still labwc; legacy X11 remains an opt-in package (`rpd-x-core`), and the old preferences apps were merged into a single Control Centre. — https://www.raspberrypi.com/news/trixie-the-new-version-of-raspberry-pi-os/
- **Official kiosk method (Wayland path).** The official tutorial autostarts the kiosk by editing `~/.config/labwc/autostart` and launching `chromium <urls> --kiosk --noerrdialogs --disable-infobars --no-first-run --start-maximized &`.
  — Raspberry Pi (official tutorial), *How to use a Raspberry Pi in kiosk mode* — https://www.raspberrypi.com/tutorials/how-to-use-a-raspberry-pi-in-kiosk-mode/. The labwc `autostart` file is part of labwc's own config set (alongside `rc.xml`, `environment`) — labwc(1) — https://labwc.github.io/labwc.1.html. The X11-era path (`/etc/xdg/lxsession/LXDE-pi/autostart`) applies only if the kiosk is switched back to X11.
- **surf under labwc.** surf "is a simple web browser based on WebKit2/GTK+ … supports the XEmbed protocol" and is steered via XProperties — X11 mechanisms — https://surf.suckless.org/. It therefore runs under XWayland (provided automatically per the labwc quote above). It works, but Chromium `--kiosk` is the path the vendor documents and tests; consider switching during hardening.
- **systemd user service with crash restart.** `Restart=on-failure` restarts "when the process exits with a non-zero exit code, is terminated by a signal … when an operation (such as service reload) times out, and when the configured watchdog timeout is triggered"; `RestartSec` "Defaults to 100ms"; restarts are "subject to unit start rate limiting configured with StartLimitIntervalSec= and StartLimitBurst=".
  — systemd.service(5) — https://man7.org/linux/man-pages/man5/systemd.service.5.html. GUI caveat: `graphical-session.target` (user manager) "is active whenever any graphical session is running. It is used to stop user services which only apply to a graphical (X, Wayland, etc.) session when the session is terminated"; graphical-only services should set `PartOf=graphical-session.target`.
  — systemd.special(7) — https://man7.org/linux/man-pages/man7/systemd.special.7.html. labwc's documented bridge: put `systemctl --user --no-block start labwc-session.target` in the labwc `autostart`/`shutdown` files — labwc(1), URL above.

  ```ini
  # ~/.config/systemd/user/dermascan-kiosk.service
  [Unit]
  Description=DermaScan kiosk
  PartOf=graphical-session.target
  After=graphical-session.target
  [Service]
  ExecStart=/home/pi/skin-cancer-project/launch_kiosk.sh
  Restart=on-failure
  RestartSec=3
  [Install]
  WantedBy=graphical-session.target
  ```

  Two conditions for boot-time start: desktop autologin must be enabled (raspi-config boot-behaviour option; user session ≠ boot), and the unit must be started from the session (the labwc-session.target line, or `systemctl --user enable` with the WantedBy above). User services do not inherit `WAYLAND_DISPLAY`/`DISPLAY` automatically — starting via the graphical-session/labwc hook is what makes the environment available. Simplest fully-official alternative: plain labwc `autostart` (no auto-restart on crash).
- **Screen blanking off.** Desktop: "Preferences > Control Centre … Display tab … Use the toggle to turn Screen Blanking on or off" (same setting exposed via raspi-config Display Options). Console: `consoleblank=` kernel parameter in `/boot/firmware/cmdline.txt` (e.g. `consoleblank=600`; `0` disables).
  — Raspberry Pi documentation, *Configuration* — https://www.raspberrypi.com/documentation/computers/configuration.html

---

## Track D — Human-participants rules for photographing students' skin

**VERDICT:** Yes — photographing students' skin as study data is human-participants research under the ISEF rulebook, and it requires constituted IRB/SRC approval **before** the first photo, plus written parental permission and minor assent for every under-18 participant. The DepEd fair pipeline (NSTF feeder memos) independently requires SRC review and a learner media-release consent.

- **Definition (it applies).** "a human participant is a living individual about whom an investigator conducting research obtains (1) data or samples through intervention or interaction with individuals(s) or (2) identifiable private information." Camera capture of students' skin is data obtained through interaction, and photos are potentially identifiable.
  — Society for Science, ISEF International Rules, *Human Participants* — https://www.societyforscience.org/isef/international-rules/human-participants/
- **Pre-approval.** "All human participant studies must be reviewed and approved by an Institutional Review Board (IRB) prior to experimentation" — before "any interaction (e.g., recruitment, data collection)". IRB = minimum three members: an educator (not the Adult Sponsor), a school administrator, and a medical or mental-health professional. — same page.
- **Minors.** "All human participant studies involving minors (students under 18 years of age) must receive assent from the student participant and written parental permission from a legal guardian." — same page.
- **Exceptions don't cover this.** The listed exempt categories (prototype testing by the student researchers only, pre-existing public data, non-interactive public observation, de-identified retrospective data) do not include photographing recruited schoolmates. — same page.
- **Forms (current 2027 rulebook numbering).** Form 1 (Checklist for Adult Sponsor), 1A (Student Checklist/Research Plan), 1B (Approval Form), Form 4 (Human Participants) + Sample Informed Consent Statement; Qualified Scientist is now Form 2B (2A = Student Support Disclosure, 2C = Regulated Research Institution) — note the renumbering from the older "Form 2". "The forms should be filled out and signed before any research takes place."
  — Society for Science, *ISEF Forms* — https://www.societyforscience.org/isef/forms/
- **Philippines / NSTF.** The national fair is governed by DepEd Memorandum No. 016, s. 2025 (*National Science and Technology Fair for SY 2024–2025*), confirmed on the official DepEd site — https://www.deped.gov.ph/2025/02/11/february-5-2025-dm-016-s-2025-national-science-and-technology-fair-for-school-year-2024-2025/ (full PDF too large to parse this session — see Open questions). A primary regional issuance in the same pipeline, DepEd Region 2 RM No. 132, s. 2025 — https://region2.deped.gov.ph/wp-content/uploads/2025/03/REGIONAL-MEMORANDUM-NO.-132-S.-2025-CONDUCT-OF-THE-NATIONAL-SCIENCE-AND-TECHNOLOGY-FAIR.pdf — lists among its enclosures "Checkpoints for SRC Review", a "Scientific Review Committee Waiver Form", a "Review and Recommendation Report", and a "Learner's Media Release Consent Form": SRC review and consent paperwork are built into the DepEd fair process.
- **Bottom line before any student is photographed:** (1) written research plan reviewed by the Adult Sponsor/mentor; (2) a properly constituted school IRB/SRC (including a medical professional, given medical imaging of minors) documents approval **before** recruitment or the first capture; (3) each participant has minor assent + written parental permission (ISEF sample consent + DepEd learner media-release consent); (4) Forms 1/1A/1B/4 (+2B if a Qualified Scientist is involved) signed pre-research; (5) the kiosk pilot stores photos per the consent terms (the "prototype testing" exemption applies only to tests run on the team members themselves).

---

## Open questions / could not verify

1. **Unique lesion count in HAM10000** — not stated in the fetched paper text; compute in-repo via `HAM10000_metadata.lesion_id.nunique()` before quoting a number in the paper.
2. **`experimental_preserve_all_tensors` in ai-edge-litert 2.1.5 specifically** — covered by the migration guide's "fully supports the TensorFlow Lite Interpreter API" parity statement and the shared interpreter source, but not independently confirmed against the 2.1.5 wheel; smoke-test on the Pi.
3. **TF RAM footprint on Pi 4** — no authoritative published figure found; only wheel size (~282 MB) and device RAM specs are citable.
4. **raspi-config desktop-autologin option** — the docs' Boot Behaviour section exists but its exact current wording wasn't extractable; verify on-device (`sudo raspi-config` → System Options).
5. **`lwrespawn` autostart wrapper** — appears only in Raspberry Pi forum posts (non-primary); not relied on above.
6. **Explicit "NSTF adopts ISEF rules" clause in a national DepEd primary source** — DM 016 s. 2025's PDF (>10 MB) could not be parsed this session, and DepEd Caraga/NCR mirrors blocked or were scan-only; the ISEF-affiliation wording is so far only in secondary reproductions (the Society's own Find-a-Fair directory lists the Philippines only in a JS dropdown, no static listing). Obtain the DM 016 s. 2025 full text from the school/division office before citing it in the paper.
7. **Blocked primary hosts** — nature.com (cookie redirect loop), freedesktop.org and sciencedirect.com (403): equivalent primaries were used instead (PMC full text, man7.org man-page mirrors, PubMed record).
