"""Markdown、Frontmatter 与浏览器可复用解析工具。"""

from __future__ import annotations

import glob
import os
import re
import shutil
from collections.abc import Sequence

_FENCED_BACKTICK_RE = re.compile(r"```.*?```", re.DOTALL)
_FENCED_TILDE_RE = re.compile(r"~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def parse_frontmatter(text: str) -> tuple[str, str]:
    """拆分 YAML frontmatter 原始块与正文；无合法闭合块时返回 ``("", text)``。"""
    if not text or not text.startswith("---"):
        return "", text
    if len(text) > 3 and text[3] not in ("\n", "\r"):
        return "", text
    end = text.find("\n---", 3)
    while end != -1:
        after = end + 4
        if after >= len(text) or text[after] in ("\n", "\r"):
            fm_block = text[3:end].strip("\r\n")
            body = text[after:].lstrip()
            return fm_block, body
        end = text.find("\n---", after)
    return "", text


def strip_frontmatter(text: str) -> str:
    """移除开头 YAML frontmatter 块，返回正文。"""
    return parse_frontmatter(text)[1]


def parse_frontmatter_list(text: str, key: str) -> list[str]:
    """提取 frontmatter 列表字段（支持 ``key: [a, b]`` 与块状 ``- item``）。"""
    fm_block, _ = parse_frontmatter(text)
    if not fm_block:
        return []
    inline = re.search(rf"^{re.escape(key)}\s*:\s*\[(.*?)\]", fm_block, re.MULTILINE)
    if inline:
        return [item.strip().strip("'\"") for item in inline.group(1).split(",") if item.strip()]
    block = re.search(
        rf"^{re.escape(key)}\s*:\s*\n((?:[ \t]*-[ \t]*.+\n?)+)", fm_block, re.MULTILINE
    )
    if block:
        items = re.findall(r"-[ \t]*(.+)", block.group(1))
        return [item.strip().strip("'\"") for item in items if item.strip()]
    return []


def strip_code_blocks(text: str) -> str:
    """移除围栏代码块（`` ``` `` / ``~~~``）及行内代码（`` `...` ``）。"""
    cleaned = _FENCED_BACKTICK_RE.sub("", text)
    cleaned = _FENCED_TILDE_RE.sub("", cleaned)
    return _INLINE_CODE_RE.sub("", cleaned)


def extract_internal_links(text: str, *, strip_code: bool = True) -> list[str]:
    """提取 Markdown 站内相对链接目标（剥离 ``#fragment``，忽略外链与纯锚点）。"""
    source = strip_code_blocks(text) if strip_code else text
    targets: list[str] = []
    for match in _MD_LINK_RE.finditer(source):
        href = match.group(1).strip()
        if not href:
            continue
        lower = href.lower()
        if lower.startswith(("http://", "https://", "mailto:", "#")):
            continue
        href = href.split("#", 1)[0].strip()
        if href:
            targets.append(href)
    return targets


def slugify_ascii(text: str, default: str = "item") -> str:
    """将文本转为小写 ASCII URL slug；无有效 ASCII 字符时返回 ``default``。"""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or default


def find_chromium_executable(extra_candidates: Sequence[str] = ()) -> str | None:
    """按环境变量、预装 Playwright 浏览器目录及 PATH 顺序查找 Chromium 可执行文件。"""
    for var in ("CHROMIUM_PATH", "CHROME", "CHROME_PATH"):
        env_path = os.environ.get(var, "").strip()
        if env_path:
            return env_path

    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    for pattern in ("chromium-*/chrome-linux*/chrome", "chromium/chrome-linux*/chrome"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[-1]

    for cand in extra_candidates:
        if cand and os.path.isfile(cand):
            return cand

    for cmd in ("chromium", "chromium-browser", "google-chrome"):
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
    return None
