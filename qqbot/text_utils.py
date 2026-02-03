import re


def convert_md_2_pure_text(md: str) -> str:
    text = md

    # ---------- 去掉强调 ----------
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # ---------- 去掉标题 ----------
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)

    # ---------- 去掉引用 ----------
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)

    # ---------- 去掉无序列表符号 ----------
    # - item, * item, + item
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # ---------- 去掉水平线 ----------
    text = re.sub(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


def delete_qq_prefix(text: str) -> str:
    return re.sub(r"^@\d+", "", text)


def delete_formatted_prefix(text: str) -> str:
    return re.sub(r"^\(\d{2}:\d{2}\)\s\d+:", "", text)

