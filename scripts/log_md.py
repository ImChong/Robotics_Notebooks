"""log.md 写入约定：新记录在文件顶部（首条 ``## [`` 之前），与首页 latest_wiki_nodes 解析一致。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "log.md"
LOG_PREAMBLE = (
    "> 核心规范：所有日常动作（ingest / query / lint / structural）必须追加记录到此文件。\n\n"
)


def prepend_log_entry(text: str, entry: str) -> str:
    """在首条 ``## [日期]`` 日志标题之前插入 entry（保留文件顶部说明行）。"""
    if not entry.endswith("\n"):
        entry = entry + "\n"
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            insert_at = i
            break
    else:
        insert_at = len(lines)
    if insert_at > 0 and lines[insert_at - 1].strip() != "" and not entry.startswith("\n"):
        entry = "\n" + entry
    return "".join(lines[:insert_at]) + entry + "".join(lines[insert_at:])


def read_log_text(log_path: Path = DEFAULT_LOG_PATH) -> str:
    if log_path.is_file():
        return log_path.read_text(encoding="utf-8")
    return LOG_PREAMBLE


def write_log_prepend(entry: str, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """将 entry 插入 log.md 顶部并写回。"""
    log_path.write_text(prepend_log_entry(read_log_text(log_path), entry), encoding="utf-8")
