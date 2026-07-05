"""Offline assistant: doctor-reviewed KB + TF-IDF retrieval + optional Gemma reword.

Design (MVP 2, locked):
- The knowledge base (``data/assistant_kb.json``) is the ONLY source of medical
  content. Entries without ``reviewed_by``/``reviewed_date`` are hidden unless
  ``SKIN_KB_DEV=1`` (doctor-review gate).
- Retrieval is a tiny pure-python TF-IDF + cosine matcher (<50 entries — no
  scikit-learn needed on the Pi).
- Gemma 3 270M via a local Ollama server may REWORD a retrieved answer or SELECT
  follow-up questions. It never authors medical facts; any failure/timeout falls
  back to the verbatim vetted text, silently.

No Streamlit imports here (mirrors tflite_shared.py) — unit-testable standalone.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

VALID_BANDS = ("benign", "pre_cancerous", "malignant", "general")
SIM_THRESHOLD = 0.25

OLLAMA_URL = os.environ.get("SKIN_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SKIN_OLLAMA_MODEL", "gemma3:270m")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KbEntry:
    id: str
    result_band: str
    questions: tuple[str, ...]
    answer: str
    always_append_disclaimer: bool = True
    reviewed_by: str | None = None
    reviewed_date: str | None = None


@dataclass(frozen=True)
class KnowledgeBase:
    fallback_answer: str
    entries: tuple[KbEntry, ...] = field(default_factory=tuple)


def _kb_path() -> Path:
    override = os.environ.get("SKIN_ASSISTANT_KB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "assistant_kb.json"


def load_kb(path: str | Path | None = None) -> KnowledgeBase:
    """Load + validate the KB. Unreviewed entries are hidden unless SKIN_KB_DEV=1."""
    p = Path(path) if path else _kb_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    dev = os.environ.get("SKIN_KB_DEV") == "1"
    entries: list[KbEntry] = []
    seen_ids: set[str] = set()
    for e in raw.get("entries", []):
        for req in ("id", "result_band", "questions", "answer"):
            if req not in e:
                raise ValueError(f"KB entry missing required field '{req}': {e.get('id', '?')}")
        if e["result_band"] not in VALID_BANDS:
            raise ValueError(f"KB entry '{e['id']}': unknown result_band '{e['result_band']}'")
        if e["id"] in seen_ids:
            raise ValueError(f"KB duplicate entry id '{e['id']}'")
        seen_ids.add(e["id"])
        if not e["questions"] or not str(e["answer"]).strip():
            raise ValueError(f"KB entry '{e['id']}': questions/answer must be non-empty")
        reviewed = bool(e.get("reviewed_by")) and bool(e.get("reviewed_date"))
        if not reviewed and not dev:
            continue  # doctor-review gate
        entries.append(
            KbEntry(
                id=e["id"],
                result_band=e["result_band"],
                questions=tuple(str(q) for q in e["questions"]),
                answer=str(e["answer"]),
                always_append_disclaimer=bool(e.get("always_append_disclaimer", True)),
                reviewed_by=e.get("reviewed_by"),
                reviewed_date=e.get("reviewed_date"),
            )
        )
    return KnowledgeBase(
        fallback_answer=str(raw.get("fallback_answer", "Please consult a health professional.")),
        entries=tuple(entries),
    )


# ── Retrieval: tiny TF-IDF + cosine (source of truth, always available) ─────────


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    n = float(len(tokens)) or 1.0
    return {t: c / n for t, c in counts.items()}


class Matcher:
    """TF-IDF over every KB question string; entry score = max over its questions."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._docs: list[tuple[KbEntry, dict[str, float]]] = []
        doc_tokens = [(e, _tokenize(q)) for e in kb.entries for q in e.questions]
        n_docs = float(len(doc_tokens)) or 1.0
        df: dict[str, float] = {}
        for _e, toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0.0) + 1.0
        self._idf = {t: math.log((1.0 + n_docs) / (1.0 + c)) + 1.0 for t, c in df.items()}
        for e, toks in doc_tokens:
            vec = {t: w * self._idf.get(t, 0.0) for t, w in _tf(toks).items()}
            self._docs.append((e, vec))

    def _vec(self, text: str) -> dict[str, float]:
        return {t: w * self._idf.get(t, 0.0) for t, w in _tf(_tokenize(text)).items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(w * b.get(t, 0.0) for t, w in a.items())
        na = math.sqrt(sum(w * w for w in a.values()))
        nb = math.sqrt(sum(w * w for w in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def match(self, query: str, band: str = "general") -> tuple[KbEntry | None, float]:
        """Best entry for ``query`` among ``band`` + general entries, or (None, score)
        when nothing clears SIM_THRESHOLD (caller shows the vetted fallback)."""
        qv = self._vec(query)
        best: KbEntry | None = None
        best_score = 0.0
        for entry, dv in self._docs:
            if entry.result_band not in (band, "general"):
                continue
            score = self._cosine(qv, dv)
            if score > best_score:
                best, best_score = entry, score
        if best_score < SIM_THRESHOLD:
            return None, best_score
        return best, best_score


def suggested_questions(kb: KnowledgeBase, band: str = "general", *, exclude_ids: set[str] | None = None, n: int = 4) -> list[tuple[str, str]]:
    """Deterministic (entry_id, first question) pairs for tappable buttons —
    band-specific entries first, then general."""
    exclude = exclude_ids or set()
    ranked = [e for e in kb.entries if e.result_band == band and e.id not in exclude]
    ranked += [e for e in kb.entries if e.result_band == "general" and e.id not in exclude]
    return [(e.id, e.questions[0]) for e in ranked[:n]]


# ── Optional Gemma 3 270M layer (Ollama). Never authors facts. ──────────────────

_REPHRASE_PROMPT = (
    "Reword the following approved medical guidance in friendly, plain language. "
    "Do not add, remove, or change any medical facts, timeframes, or recommendations. "
    "Reply with the reworded text only.\n\nApproved text: {text}"
)

# Timeframe substrings that must survive a reword (guard against fact-dropping).
_TIMEFRAME_RE = re.compile(r"\b(within\s+)?\d+(-\d+)?\s+(week|weeks|month|months|day|days)\b", re.IGNORECASE)


class OllamaRephraser:
    """Thin localhost Ollama client. Every failure path returns the verbatim text."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health(self) -> dict:
        try:
            import requests

            r = requests.get(f"{self.base_url}/api/tags", timeout=1)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            ready = any(m.startswith(self.model.split(":")[0]) for m in models)
            return {"status": "ok" if ready else "missing_model", "models": models}
        except Exception as exc:  # noqa: BLE001 — any failure means "not available"
            return {"status": "error", "reason": str(exc)}

    def _generate(self, prompt: str, *, num_predict: int) -> str | None:
        try:
            import requests

            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": 1024},
                },
                timeout=(2, 10),
            )
            r.raise_for_status()
            return str(r.json().get("response", "")).strip()
        except Exception:  # noqa: BLE001 — silent fallback is the contract
            return None

    def reword(self, approved_text: str) -> tuple[str, bool]:
        """Return (text, ai_reworded). Falls back to the verbatim approved text when
        the model is unavailable or its output fails the fact-preservation guards."""
        out = self._generate(_REPHRASE_PROMPT.format(text=approved_text), num_predict=120)
        if not out:
            return approved_text, False
        if len(out) > 2.5 * len(approved_text):
            return approved_text, False
        # Every timeframe in the approved text must survive the reword.
        for m in _TIMEFRAME_RE.finditer(approved_text):
            frag = m.group(0).lower()
            if frag not in out.lower():
                return approved_text, False
        return out, True

    def pick_followups(self, questions: list[str], n: int = 3) -> list[str] | None:
        """Ask Gemma to SELECT question indices (never author text). None on any
        parse failure — caller falls back to suggested_questions()."""
        if not questions:
            return None
        listing = "\n".join(f"{i}: {q}" for i, q in enumerate(questions))
        out = self._generate(
            "From this numbered list of questions, reply with only the numbers of the "
            f"{min(n, len(questions))} most useful follow-up questions, comma-separated.\n{listing}",
            num_predict=20,
        )
        if not out:
            return None
        try:
            idxs = [int(x) for x in re.findall(r"\d+", out)][:n]
        except ValueError:
            return None
        if not idxs or any(i < 0 or i >= len(questions) for i in idxs):
            return None
        return [questions[i] for i in dict.fromkeys(idxs)]
