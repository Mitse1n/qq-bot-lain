from types import SimpleNamespace

from qqbot.message_parsing import (
    parse_message_content,
    parse_message_content_from_history,
)
from qqbot.text_utils import (
    convert_md_2_pure_text,
    delete_formatted_prefix,
    delete_qq_prefix,
)


def test_parse_message_content_converts_known_mentions_only():
    known_users = {"12345"}
    text = "hi @12345 and @99999"

    segments = parse_message_content(text, known_users)

    assert segments == [
        {"type": "text", "data": {"text": "hi "}},
        {"type": "at", "data": {"qq": "12345"}},
        {"type": "text", "data": {"text": " and @99999"}},
    ]


def test_parse_message_content_from_history():
    history = [SimpleNamespace(user_id="12345"), SimpleNamespace(user_id="67890")]
    text = "hello @67890"

    segments = parse_message_content_from_history(text, history)

    assert segments == [
        {"type": "text", "data": {"text": "hello "}},
        {"type": "at", "data": {"qq": "67890"}},
    ]


def test_convert_md_2_pure_text_strips_basic_markdown():
    md = "# Title\n\n- **bold** _italic_\n> quote\n---\ntext"
    assert convert_md_2_pure_text(md) == "Title\nbold italic\nquote\n\ntext"


def test_delete_qq_prefix_only_at_start():
    assert delete_qq_prefix("@12345 hello") == " hello"
    assert delete_qq_prefix("hi @12345") == "hi @12345"


def test_delete_formatted_prefix_at_start():
    assert delete_formatted_prefix("(12:34) 12345:hello") == "hello"
    assert delete_formatted_prefix("x(12:34) 12345:hello") == "x(12:34) 12345:hello"
