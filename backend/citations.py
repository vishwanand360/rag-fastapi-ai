"""Validate generated source labels against page metadata actually retrieved."""
from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[Source:\s*(.+?),\s*p\.\s*(\d+)\]")
_CITATION_MARKER_RE = re.compile(r"\[Source:")
_ABSTENTION_RE = re.compile(
    r"\b(?:i (?:cannot|can't) answer|insufficient (?:information|context)|"
    r"the (?:provided )?(?:documents|excerpts|context) (?:do not|don't|does not) "
    r"(?:support|contain|provide))\b",
    re.IGNORECASE,
)


def citations_for_answer(answer: str, contexts: list[dict]) -> list[dict]:
    """Return cited source/page pairs, rejecting labels absent from retrieved context."""
    matches = list(_CITATION_RE.finditer(answer))
    if len(matches) != len(_CITATION_MARKER_RE.findall(answer)):
        raise ValueError("Answer contains a malformed source citation")
    if not matches and answer.strip() and not _ABSTENTION_RE.search(answer):
        raise ValueError("Answer has no verifiable page citation")

    context_by_label: dict[tuple[str, int], dict] = {}
    for context in contexts:
        try:
            key = (str(context["source"]), int(context["page"]))
        except (KeyError, TypeError, ValueError):
            continue
        context_by_label.setdefault(key, context)

    cited: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for match in matches:
        label = (match.group(1).strip(), int(match.group(2)))
        context = context_by_label.get(label)
        if context is None:
            raise ValueError("Answer cites a source or page that was not retrieved")
        if label not in seen:
            cited.append({"source": label[0], "page": label[1], "chunk_id": context.get("id")})
            seen.add(label)
    return cited
