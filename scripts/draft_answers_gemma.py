"""Draft the Questions-tab answers with Gemma 4, on the PC, into dermascan/answers.json.

This is how "Gemma 4" takes part in EPIVUE without running on the Pi: the Pi has
4 GB and the smallest Gemma 4 (E2B) wants 8, so the model drafts at build time
here, and the kiosk serves the drafts offline, instantly, with the badge
"DRAFTED BY GEMMA 4 · NOT YET REVIEWED BY A DOCTOR" until a health professional
fills `reviewed_by` and `reviewed_date`.

    ollama pull gemma4:e4b                      # once, on the PC (any gemma4 tag works)
    venv/bin/python scripts/draft_answers_gemma.py --dry-run     # show the drafts, write nothing
    venv/bin/python scripts/draft_answers_gemma.py               # rewrite answers.json
    venv/bin/python scripts/draft_answers_gemma.py --model gemma4:e2b --only urgent_what_now

What the model is asked to do, and nothing more: say the SAME thing in plainer
words. The approved answer is the source; the timeframes come from
dermascan/verdict.py (ACTION); the rules are the ones tests/test_answers.py
enforces. Every draft is checked before it is kept:

    - no clinical or model jargon (the same word list as the tests)
    - no digits and no percentages
    - every timeframe of the approved answer is still there, word for word
    - at most MAX_WORDS words, and not shorter than a third of the original
    - never the words "safe" / "harmless" / "nothing to worry" unless the original had them

A draft that fails is dropped and the approved answer stays, with a line saying why.
The questions (the phrasings a visitor can tap) are never rewritten: the TF-IDF
match in dermascan/answers.py was tuned on them.

The Pi never runs this. requirements.txt is untouched: urllib talks to Ollama.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dermascan import verdict  # noqa: E402
from dermascan.answers import ANSWERS_PATH  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b"
MAX_WORDS = 70
JARGON = ("malignant", "benign", "softmax", "inconclusive", "pre_cancerous", "classifier", "model", "tflite", "lesion", "dermoscopy", "biopsy")
REASSURANCE = ("safe", "harmless", "nothing to worry", "no need to worry", "don't worry", "do not worry")

PROMPT = """You are rewording approved patient-facing text for a skin-check kiosk in a barangay health centre.
Say the SAME thing in plainer, friendlier English for someone with no medical background.
Rules, all of them:
- Keep every fact, every timeframe and every recommendation exactly as they are. Do not add any.
- Use everyday words. Never use: {jargon}.
- No numbers, no percentages, no bullet points, no headings.
- Never say or imply that a spot is safe or harmless.
- At most {max_words} words. Reply with the reworded text only, nothing else.

The visitor's question: {question}
Approved text to reword: {answer}"""


def timeframes(text: str) -> list[str]:
    """The timeframe phrases the approved text promises, which a draft must keep."""
    found = [a.lower() for a in (verdict.ACTION["pre_cancerous"], verdict.ACTION["malignant"]) if a.lower().rstrip(".") in text.lower()]
    found += re.findall(r"within (?:a|one|two|three|a few) (?:to \w+ )?(?:day|week|month)s?", text.lower())
    return sorted(set(found))


def check_draft(draft: str, original: str) -> str | None:
    """None when the draft may replace the original, else the reason it may not."""
    low = draft.lower()
    for w in JARGON:
        if re.search(rf"\b{re.escape(w)}\b", low):
            return f"uses '{w}'"
    if re.search(r"\d", draft) or "%" in draft:
        return "has a number"
    for t in timeframes(original):
        if t.rstrip(".") not in low:
            return f"drops the timeframe '{t}'"
    for r in REASSURANCE:
        if r in low and r not in original.lower():
            return f"adds reassurance '{r}'"
    words = len(draft.split())
    if words > MAX_WORDS:
        return f"{words} words, over {MAX_WORDS}"
    if words < max(8, len(original.split()) // 3):
        return "too short to say the same thing"
    if "\n" in draft.strip() or draft.lstrip().startswith(("-", "*", "#")):
        return "not one plain paragraph"
    return None


def ask_ollama(model: str, prompt: str, timeout: float = 120.0) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 200}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["response"].strip().strip('"')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="an Ollama tag; `ollama list` shows what is pulled")
    ap.add_argument("--only", nargs="*", default=None, help="entry ids to draft (default: every entry not yet reviewed)")
    ap.add_argument("--dry-run", action="store_true", help="print the drafts and write nothing")
    a = ap.parse_args()

    raw = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    stamp = f"{a.model.split(':')[0].replace('gemma', 'Gemma ')} {a.model.split(':')[1] if ':' in a.model else ''}".strip() + f", {date.today().isoformat()}"
    kept = dropped = 0
    for e in raw["entries"]:
        if a.only is not None and e["id"] not in a.only:
            continue
        if e.get("reviewed_by") and e.get("reviewed_date"):
            print(f"{e['id']}: reviewed by {e['reviewed_by']}, left as is")
            continue
        try:
            draft = ask_ollama(a.model, PROMPT.format(jargon=", ".join(JARGON), max_words=MAX_WORDS, question=e["questions"][0], answer=e["answer"]))
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            raise SystemExit(f"Ollama did not answer ({exc}). Is it running, and is {a.model} pulled?") from exc
        why = check_draft(draft, e["answer"])
        if why:
            dropped += 1
            print(f"{e['id']}: draft dropped ({why})")
            continue
        kept += 1
        print(f"{e['id']}:\n  was: {e['answer']}\n  now: {draft}")
        e["answer"], e["drafted_by"] = draft, stamp
    print(f"\n{kept} drafted, {dropped} dropped")
    if a.dry_run or not kept:
        return
    raw["note"] = raw["note"].split(" Drafts:")[0] + f" Drafts: answers marked drafted_by were reworded by {stamp} with scripts/draft_answers_gemma.py and checked by the team; a doctor's review still clears the badge."
    ANSWERS_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {ANSWERS_PATH}. Now run: venv/bin/python -m pytest -q tests/test_answers.py")


if __name__ == "__main__":
    main()
