"""Guardrails around the agent: what a task may be, what it is told about text others wrote, and
what an answer must satisfy before anyone reads it. The fence, its note and credential redaction
are `underhood.tools.guard`'s, shared with the fine-tunes; fencing a tool's output is
`tools.execute_tool`'s."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection

from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from underhood.tools.guard import REDACTED, SECRETS, UNTRUSTED_NOTE, redact

from aigent.config import MAX_ANSWER_CHARS, MAX_TASK_CHARS
from aigent.errors import GuardrailTripped

WITHHELD = "[answer withheld: it contained something shaped like a credential]"

PASSAGE_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*#\d{4}\b")
BRACKETS = re.compile(r"\[([^\[\]]*)\]")  # a citation is an id in square brackets, as asked


def check_task(text: str) -> str:
    """The task as the agent will see it, or `GuardrailTripped` before anything is sent."""
    cleaned = text.strip()
    if not cleaned:
        raise GuardrailTripped("the task is empty")
    if len(cleaned) > MAX_TASK_CHARS:
        raise GuardrailTripped(f"the task is {len(cleaned):,} characters, over {MAX_TASK_CHARS:,}")
    controls = {ch for ch in cleaned if unicodedata.category(ch) == "Cc" and ch not in "\n\t"}
    if controls:
        raise GuardrailTripped(f"the task carries control characters {sorted(map(ord, controls))}")
    if any(secret.search(cleaned) for secret in SECRETS):
        raise GuardrailTripped("the task contains something shaped like a credential")
    return cleaned


class Answer(BaseModel):
    """A final answer fit to show: within its cap, citing only retrieved passages, leaking no
    credential. Built by `check_answer`, one field per rule so every rule reports."""

    text: str
    cites: list[str]
    leaks: list[str]

    @field_validator("text")
    @classmethod
    def _bounded(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("the answer is empty")
        if len(text) > MAX_ANSWER_CHARS:
            raise ValueError(f"the answer is {len(text):,} characters, over {MAX_ANSWER_CHARS:,}")
        return text

    @field_validator("cites")
    @classmethod
    def _cites_what_was_found(cls, cites: list[str], info: ValidationInfo) -> list[str]:
        found = set((info.context or {}).get("found", ()))
        invented = sorted(set(cites) - found)
        if invented:
            raise ValueError(f"cites passages it never retrieved: {', '.join(invented)}")
        return cites

    @field_validator("leaks")
    @classmethod
    def _leaks_nothing(cls, leaks: list[str]) -> list[str]:
        if leaks:
            raise ValueError(f"contains {len(leaks)} credential-shaped string(s)")
        return leaks


def cited(text: str) -> list[str]:
    """Every passage id `text` cites: ids inside square brackets, not ids it merely mentions."""
    return [id_ for group in BRACKETS.findall(text) for id_ in PASSAGE_ID.findall(group)]


def check_answer(text: str, found: Collection[str]) -> list[str]:
    """Every rule `text` breaks, as messages; empty when it is fit to show."""
    leaks = [match[0] for secret in SECRETS for match in secret.finditer(text)]
    try:
        Answer.model_validate(
            {"text": text, "cites": cited(text), "leaks": leaks},
            context={"found": found},
        )
    except ValidationError as failed:
        return [str(error["msg"]).removeprefix("Value error, ") for error in failed.errors()]
    return []


def shown(text: str, violations: Collection[str]) -> str:
    """The answer as a reader gets it: withheld when it leaks a credential, else as written."""
    return WITHHELD if any("credential" in violation for violation in violations) else text


__all__ = ["REDACTED", "SECRETS", "UNTRUSTED_NOTE", "redact"]
