import re
from typing import Iterable, List, Protocol, Set


class HasUserId(Protocol):
    user_id: str


def known_user_ids(history: Iterable[HasUserId]) -> Set[str]:
    return {str(msg.user_id) for msg in history}


def parse_message_content(text: str, known_users: Set[str]) -> List[dict]:
    """Parse text to convert @mentions to at segments."""
    segments: List[dict] = []
    last_end = 0
    pattern = re.compile(r"@(\d{5,})")

    for match in pattern.finditer(text):
        qq_id = match.group(1)
        start, end = match.span()

        if qq_id in known_users:
            if start > last_end:
                segments.append(
                    {"type": "text", "data": {"text": text[last_end:start]}}
                )
            segments.append({"type": "at", "data": {"qq": qq_id}})
            last_end = end

    if last_end < len(text):
        segments.append({"type": "text", "data": {"text": text[last_end:]}})

    if not segments:
        segments.append({"type": "text", "data": {"text": ""}})

    return segments


def parse_message_content_from_history(text: str, history: Iterable[HasUserId]) -> List[dict]:
    return parse_message_content(text, known_user_ids(history))

