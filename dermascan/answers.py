"""The Questions tab: a visitor asks, the kiosk answers from a fixed bank of written answers.

There is no language model here, on purpose. Every answer a visitor can read
is written out in answers.json, so nothing medical is ever made up on the spot,
and a match takes well under a millisecond on the Pi.

    question (tapped or typed)
      -> TF-IDF over every phrasing in answers.json   (pure Python, ~25 entries)
      -> best entry for this result, or for everyone   ("general")
      -> its answer, plus the visitor's own sign lines if it is about their spot
      -> below ANSWER_MATCH_MIN: the fallback ("ask a health worker")

An answer is "reviewed" only when a health professional has filled in
reviewed_by and reviewed_date in answers.json. Until then the page says it was
written by the project team. Nothing is hidden for being unreviewed: the badge
tells the truth instead.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dermascan import config

ANSWERS_PATH = Path(__file__).resolve().parent / "answers.json"
BANDS = ("clean", "checked", "urgent", "unsure", "general")

# Which answers fit which verdict (dermascan/verdict.py states).
BAND_FOR_STATE = {
    "low_concern": "clean",
    "needs_attention": "checked",
    "uncertain_caution": "checked",
    "urgent": "urgent",
    "uncertain": "unsure",
    "refused": "unsure",
    "error": "unsure",
}

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Entry:
    id: str
    band: str
    questions: tuple[str, ...]
    answer: str
    about_spot: bool = False
    reviewed_by: str | None = None
    reviewed_date: str | None = None

    @property
    def reviewed(self) -> bool:
        return bool(self.reviewed_by and self.reviewed_date)


# Under every answer, so a visitor knows who wrote it.
BADGE_UNREVIEWED = "WRITTEN BY THE PROJECT TEAM · NOT YET REVIEWED BY A DOCTOR"
BADGE_REVIEWED = "REVIEWED BY {name}"


@dataclass(frozen=True)
class Answer:
    question: str  # what the visitor asked, as they asked it
    text: str
    entry_id: str | None  # None means the fallback
    reviewed_by: str | None
    suggestions: list[str]

    def to_dict(self) -> dict:
        badge = BADGE_REVIEWED.format(name=self.reviewed_by.upper()) if self.reviewed_by else BADGE_UNREVIEWED
        return dict(self.__dict__, reviewed=bool(self.reviewed_by), badge=badge)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class AnswerBank:
    """answers.json, loaded and checked once, with a TF-IDF index over every phrasing."""

    def __init__(self, path: Path = ANSWERS_PATH) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.fallback: str = raw["fallback_answer"]
        self.entries: list[Entry] = []
        for e in raw["entries"]:
            if e["band"] not in BANDS:
                raise ValueError(f"answers.json: {e['id']} has unknown band {e['band']!r}")
            if not e["questions"] or not e["answer"].strip():
                raise ValueError(f"answers.json: {e['id']} needs questions and an answer")
            self.entries.append(Entry(
                e["id"], e["band"], tuple(e["questions"]), e["answer"], bool(e.get("about_spot")),
                e.get("reviewed_by"), e.get("reviewed_date"),
            ))
        if len({e.id for e in self.entries}) != len(self.entries):
            raise ValueError("answers.json: two entries share an id")
        # TF-IDF: every phrasing is a document; an entry scores its best phrasing.
        docs = [(e, Counter(_words(q))) for e in self.entries for q in e.questions]
        df = Counter(w for _, words in docs for w in words)
        self._idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
        self._docs = [(e, self._vector(words)) for e, words in docs]

    def _vector(self, words: Counter) -> dict[str, float]:
        """Term frequency x inverse document frequency, for one bag of words."""
        total = sum(words.values()) or 1
        return {w: n / total * self._idf.get(w, 0.0) for w, n in words.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(v * b.get(w, 0.0) for w, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def match(self, question: str, band: str = "general") -> tuple[Entry | None, float]:
        """Best entry among this band's and the general ones, or (None, score) below the floor."""
        q = self._vector(Counter(_words(question)))
        best, best_score = None, 0.0
        for entry, vec in self._docs:
            if entry.band not in (band, "general"):
                continue
            score = self._cosine(q, vec)
            if score > best_score:
                best, best_score = entry, score
        return (best, best_score) if best_score >= config.ANSWER_MATCH_MIN else (None, best_score)

    def suggestions(self, band: str = "general", exclude: str | None = None) -> list[str]:
        """Questions to tap: this result's own first, then one about the visitor's spot, then the rest.

        With no result yet ("general"), questions about "my spot" go last: there is no spot.
        """
        own = [e for e in self.entries if e.band == band and band != "general"]
        spot = [e for e in self.entries if e.band == "general" and e.about_spot]
        rest = [e for e in self.entries if e.band == "general" and not e.about_spot]
        n = config.SUGGESTED_QUESTIONS
        ranked = rest + spot if band == "general" else own[: n - 1] + spot[:1] + own[n - 1 :] + spot[1:] + rest
        return [e.questions[0] for e in ranked if e.id != exclude][:n]

    def ask(self, question: str, state: str | None = None, sign_lines: list[str] | None = None) -> Answer:
        """The answer to `question` for a visitor whose result is `state` (a verdict state, or None)."""
        sign_lines = [line.strip() for line in sign_lines or [] if line.strip()]
        band = BAND_FOR_STATE.get(state or "", "general")
        entry, _ = self.match(question, band)
        if entry is None:
            return Answer(question, self.fallback, None, None, self.suggestions(band))
        text = entry.answer
        if entry.about_spot and sign_lines:
            text += " On your photo: " + "; ".join(line[0].lower() + line[1:] for line in sign_lines) + "."
        return Answer(question, text, entry.id, entry.reviewed_by if entry.reviewed else None, self.suggestions(band, exclude=entry.id))


@lru_cache(maxsize=1)
def get_bank() -> AnswerBank:
    return AnswerBank()
