# Aligning the paper with the device — notes for the team

For *EPIVUE: Gemma 4-Driven Artificial Intelligence and Convolutional Neural Network (CNN)
Framework for Detecting Early Signs of Skin Cancer* (Amor, Dueñas, Mande; adviser Casiple;
108-page PDF of 29 October 2026). Page numbers are PDF pages. Each item says what the
paper claims, what the kiosk on `main` actually does, and the wording to use. The code
changed on 29 October 2026 (branch `feat/epivue-align`) so that every claim below is
true of the device; the paper has to change where it claims more than that.

Read `AGENTS.md` and `docs/METRICS.md` first: the numbers and the stack decisions are
there, with dates.

## 1. Must change — the claim is not true of the device

| # | Paper says (page) | The device | Write instead |
|---|---|---|---|
| 1 | **Grad-CAM heatmaps** (objectives p3/p36, abstract p25, framework p37, block diagram p42, survey p79) | TensorFlow Lite on the Pi has no gradients, so Grad-CAM is impossible there. The kiosk shows **Eigen-CAM**: the first principal component of EfficientNetB0's last 7×7×1280 feature map, read from the deployed `.tflite` after one forward pass (`dermascan/cam.py`, Muhammad & Yeasin 2020). It is class-agnostic (where the network looked, not "why cancer"), staff-only, built on demand (~1 s), never on the scan path. | "a class-activation heatmap (Eigen-CAM, a gradient-free method suited to TensorFlow Lite on the Raspberry Pi) shown to staff beside the photo" |
| 2 | **Gemma 4 chatbot on the device** (title; objective 4; "Recommendations were then given by the integrated Gemma4", p58–60, p67) | The Pi is a 4 GB board; Gemma 4's smallest model (E2B, ~4 GB) needs 8 GB, and the team's own logbook notes the hardware limit (p106). Gemma 4 **drafts the Questions-tab answers at build time on the PC** (`scripts/draft_answers_gemma.py`, Ollama) from the team's approved answers and the verdict timeframes; the kiosk serves them offline with the badge "DRAFTED BY GEMMA 4 · NOT YET REVIEWED BY A DOCTOR". Nothing is generated on the device. | "a question-and-answer tab whose answers were drafted with Gemma 4 on the development PC, checked by the team, and served offline on the device; no text is generated at the kiosk" |
| 3 | **UV fluorescence imaging** and "multi-modal sensing" (p2, p12, p37) | The hardware is one RGB camera and a white LED ring. | Remove. |
| 4 | **Mobile application on Railway** (p10, p54) | Railway hosts a web demo: the same kiosk page in web mode (nothing kept, no camera needed). It is a website, not an app, and the Pi does not use it. | "a public web demo of the same interface (Railway), for review; the kiosk itself works offline" |
| 5 | **"Biopsy report" / "biopsy-confirmed" ground truth** (p68–71) | The test images were ISIC archive images; the reference is ISIC's `diagnosis` metadata, and two of the nine (ISIC_0000108, ISIC_0000032) carry `concomitant_biopsy: false`. | "compared with the ISIC archive's recorded diagnosis" |
| 6 | **SCC detection** (p34, p65) | HAM10000 has no SCC class; SCC in situ is folded into `akiec`, which the kiosk maps to *pre-cancerous*. The malignant class is melanoma + BCC. | Drop SCC, or say "actinic keratosis / Bowen's disease (pre-cancerous)". |
| 7 | **"standardizes diagnosis by eliminating subjective human error"** (p64); "real-time diagnostic recommendations" (p55); biopsy, re-examination and "targeted topical therapies" as outputs (p66) | Screening only. The device never diagnoses or prescribes; every visitor line is "see a skin doctor within …" or "nothing stood out, keep an eye on it" (`dermascan/verdict.py`, no green all-clear). | "a screening aid that tells the visitor whether, and how soon, to show the spot to a health worker" |
| 8 | **Diameter (D)** in the ABCDE rule (p3, p9, p53) | No ruler, no scale: D is "size in frame" (share of the photo), shown to staff only with "NO SCALE". The paper's own UI sketch already says "Size in frame" (p106). | "D is shown as size in the frame; millimetres need a scale the camera does not have" |
| 9 | **Single-image percentages as results** (Table 2 p62–64, panels p68–72: 100 %, 98.9 %…) and the promised accuracy/precision/recall (p13, p58) never reported | `docs/METRICS.md` has the real numbers: on the frozen external bench (602 photos from other hospitals) the October model flags 121/149 cancers (sensitivity 0.81) with benign specificity 0.59; on the 151 camera-style close-ups it flags 28/31 cancers. The decision rule is `cancer score ≥ 0.47`, not the tallest bar. | Report sensitivity and specificity from METRICS.md with the bench's size and source; keep single images only as illustrations |
| 10 | **Label mismatch** in Fig. 25 (p71: "Pre-cancerous 100.0 %" labelled "Benign"), admitted as a display bug (p72) | Fixed: the staff drawer prints the decision line `cancer score … · limit … → flagged / not flagged`, and the displayed label is the rule's label, never the tallest bar. | Note the fix, or drop the figure. |

## 2. Should change — true but incomplete or inconsistent

- **A photo check exists and the paper never mentions it** (no gate anywhere). Before the
  model runs, the kiosk checks skin / focus / light / one spot in the ring, and labels the
  photo: *SPOT FOUND*, *NOT A SPOT · PLAIN SKIN*, *NOT A SPOT · SEVERAL MARKS*, *NOT SKIN*,
  *SPOT · NOT IN THE RING*, *CAN'T READ · BLURRY* … It draws its outline on the photo so the
  visitor sees what it picked out. On "NOT A SPOT" the model's numbers are withheld from the
  visitor (the model has no "not a lesion" class, so they would mean nothing). This is a
  method step and belongs in Methods and in Fig. 10. Its limits are heuristics measured on
  dermoscopy, not yet on Pi captures (`config.py`, lines marked GUESSED).
- **ABCDE is computed, not "learned by the CNN during training"** (p53, p58). A = fold
  misfit, B = perimeter²/(4π·area), C = colour groups in CIE Lab, on the photo check's
  outline (`dermascan/signs.py`, ~10 ms). They are shown to help a person look and never
  change the verdict. E is a side-by-side of the first and latest saved photo for a person
  to judge: no automatic "changed" (on 71 same-mole pairs any limit under 15 % false
  "changed" caught under 5 % of real change, METRICS.md).
- **Speed:** "within 1 second" (p68–71) vs "2 s / 3 s / 2 s" (p67, p72). The honest number is
  the kiosk's own log line (`scan status=ok … total=… ms`), typically 2–3 s on the Pi 4 with
  four-view averaging; quote that.
- **Dataset:** "fine-tuned on HAM10000" (p42) is the June model. The deployed October 2026
  model also trained on ISIC collections BCN20000, MSK-1..5, MILK10k and PAD-UFES-20, split
  by lesion (`training/prepare_datasets.py`), because the HAM10000-only model missed a
  quarter of the cancers from other hospitals (METRICS.md, 18 Oct). Say so, and say the
  class mapping: `nv, bkl, df, vasc → benign`, `akiec → pre-cancerous`, `mel, bcc → malignant`.
- **Possible test leakage** (p62–64): ISIC_0028063, ISIC_0029930 and ISIC_0031228 are
  HAM10000 IDs. Unless they were in the held-out split, they are training images. Use
  images from the external bench (`training/external_bench.csv`) for examples.
- **"Users" vs staff:** the kiosk has a visitor side (plain words, no percentages) and a
  staff side behind a button (bars, cancer score, A B C D E numbers, heatmap, gate numbers).
  The paper's screenshots mix the two; label each as visitor or staff.
- **Arithmetic / IDs:** "0.5 % benign + 95.0 % malignant" (p60) vs Table 2's 5.0 %;
  "B1.7 %" (p63); Fig. 20's ISIC_0000142 vs the table's ISIC_0000144.
- **Duplicated introduction** paragraphs on p34–36.
- **Saved scans:** they live on the SD card in `~/.dermascan/saved/` (photo + numbers, no
  name) until staff press "End event — erase all"; "clear cache" in the sketch (p105) is
  that button. Nothing is sent anywhere (`docs/PRIVACY.md`).

## 3. Already true — keep, and cite the code

- EfficientNetB0, ImageNet weights, 224×224 input, 3-class softmax head, split by
  `lesion_id`, exported to TensorFlow Lite for the Pi 4 (`training/`, `models/`).
- Three-class result with a percentage per class — staff drawer, calibrated by temperature
  scaling for display; the decision uses raw probabilities and `thresholds.json`.
- Urgent banner **"SEE A DOCTOR SOON"** / "See a skin doctor within one to two weeks";
  pre-cancerous **"GET THIS CHECKED"** / "within a month"; benign **"NOTHING STOOD OUT"**
  with the safety line, never green (`dermascan/verdict.py`).
- "Ask about your result" with tappable questions; Home · New check · Saved · Questions ·
  Settings; upload as well as the live camera; the "NOT A DIAGNOSIS" strip on every screen.
- Offline: no network on the scan path. The one exception is the staff "Update from GitHub"
  button, which runs outside the kiosk process.

## 4. Figures to redo

- Block diagram (p42): Input → **Photo check (label + outline)** → EfficientNetB0 (TFLite)
  → Decision (cancer score ≥ limit) → Visitor words / Staff drawer (bars, A B C D E,
  Eigen-CAM) → Questions (answers drafted with Gemma 4 on the PC, served offline).
- Fig. 10 (p54): the check screen with its chip and outline.
- Conceptual framework (p37): remove UV imaging; "AI-generated recommendations" →
  "pre-written answers".
