# DermaScan Assistant (MVP 2)

Offline Q&A assistant on the Results/Assistant screens. Architecture:

```
User question (tap a suggested question, or type on PC)
  → TF-IDF retrieval over doctor-reviewed answers   [always on — source of truth]
  → Gemma 3 270M via Ollama rewords the approved answer   [optional toggle]
  → answer + disclaimer shown; badge says "AI-reworded" or "Verbatim vetted text"
```

**Safety model:** the LLM never authors medical content. It may only (a) reword a
retrieved doctor-approved answer and (b) select follow-up questions from the KB.
Guards reject rewords that drop timeframes ("within 1-2 weeks") or bloat the text;
any Ollama failure/timeout silently falls back to the verbatim vetted answer.

## Knowledge base & doctor review workflow

- File: `data/assistant_kb.json`. Each entry: `id`, `result_band`
  (`benign|pre_cancerous|malignant|general`), `questions[]` (paraphrases the
  matcher trains on — more is better), `answer`, `always_append_disclaimer`,
  `reviewed_by`, `reviewed_date`.
- **Entries with empty `reviewed_by`/`reviewed_date` are hidden at runtime.**
  Ship nothing a clinician hasn't signed off. For development/demos, set
  `SKIN_KB_DEV=1` to show unreviewed drafts.
- Review process: hand the JSON (or a rendered copy) to the supervising
  doctor/adviser; they edit `answer` text and fill `reviewed_by` + `reviewed_date`.
- Keep answers consistent with `backend/recommendations.py` action strings.

## Ollama + Gemma setup

### Raspberry Pi 4 (kiosk)

```bash
bash scripts/setup_ollama_pi.sh   # the ONLY online step; offline afterwards
```

Installs Ollama, applies low-RAM systemd limits (1 model, 1 parallel request,
1024 ctx), pulls `gemma3:270m` (~300MB resident when loaded).

- **RAM:** fits a 2GB Pi alongside Streamlit + TFLite, but 4GB is recommended.
- **Latency:** expect ~3–8s per reword on the Pi 4 CPU; the client timeout is
  10s and falls back to verbatim text, so slow answers never block the UI.

### PC / WSL (development)

Install Ollama for your OS (https://ollama.com), then `ollama pull gemma3:270m`.
The app auto-detects it; the Settings toggle "AI wording (Gemma via Ollama)"
shows 🟢/🟡/⚪ status.

Env knobs: `SKIN_OLLAMA_URL` (default `http://127.0.0.1:11434`),
`SKIN_OLLAMA_MODEL` (default `gemma3:270m`), `SKIN_ASSISTANT_KB` (KB path),
`SKIN_KB_DEV=1` (show unreviewed drafts).

## Why not a full Gemma chatbot?

The device must run offline on a Pi 4 (2–4GB). Models ≥1B swap to SD card
(seconds per token, OOM risk next to the CNN); 270M is coherent enough to reword
but not reliable enough to author medical guidance. Retrieval over vetted answers
gives zero-hallucination responses with chatbot-like interaction — a deliberate
safety choice, not a fallback. See `tests/test_assistant.py` for the guard suite.
