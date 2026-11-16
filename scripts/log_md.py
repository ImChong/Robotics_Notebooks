"""log.md 写入约定：新记录在文件顶部（首条 ``## [`` 之前）。叙事层，站点活动以 git 为准。"""

from __future__ import annotations

import re
import secrets
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "log.md"
LOG_PREAMBLE = (
    "> 核心规范：ingest / query 的意图与结论建议记录到此文件（叙事层）。"
    "站点活动（首页最新节点 / 更新记录）以 git 为准。\n\n"
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


# ── log.d/ 碎片：PR 只新增碎片文件，不直接改 log.md（避免并行 PR 在文件顶部同一处冲突）──
# main 上的 export.yml 调用 fold_log_fragments() 把碎片按文件名顺序并入 log.md 顶部并删除碎片。
DEFAULT_FRAGMENT_DIR = REPO_ROOT / "log.d"
FRAGMENT_README = "README.md"


def write_log_fragment(
    entry: str, op: str, fragment_dir: Path = DEFAULT_FRAGMENT_DIR, today: str | None = None
) -> Path:
    """把一条日志写成 ``log.d/<日期>-<时分秒>-<op>-<随机>.md``，返回路径。

    文件名唯一，并行 PR 各自新增文件，互不冲突；按文件名排序即时间顺序。
    """
    now = datetime.now()
    day = today or now.date().isoformat()
    name = f"{day}-{now.strftime('%H%M%S')}-{op}-{secrets.token_hex(3)}.md"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    path = fragment_dir / name
    if not entry.endswith("\n"):
        entry += "\n"
    path.write_text(entry, encoding="utf-8")
    return path


def list_log_fragments(fragment_dir: Path = DEFAULT_FRAGMENT_DIR) -> list[Path]:
    if not fragment_dir.is_dir():
        return []
    return sorted(p for p in fragment_dir.glob("*.md") if p.name != FRAGMENT_README)


def fold_log_fragments(
    log_path: Path = DEFAULT_LOG_PATH, fragment_dir: Path = DEFAULT_FRAGMENT_DIR
) -> list[Path]:
    """把全部碎片并入 log.md 顶部（最新在上）并删除碎片；返回已并入的碎片路径。"""
    fragments = list_log_fragments(fragment_dir)
    if not fragments:
        return []
    text = read_log_text(log_path)
    # 旧→新依次 prepend，最后插入的（最新）落在最上面
    for frag in fragments:
        entry = frag.read_text(encoding="utf-8").strip("\n") + "\n\n"
        text = prepend_log_entry(text, entry)
    log_path.write_text(text, encoding="utf-8")
    for frag in fragments:
        frag.unlink()
    return fragments


_LOG_DATE_HEADING_RE = re.compile(r"^## \[(\d{4})-(\d{2})-\d{2}\]")
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "docs" / "plans" / "archive"


def _month_index(year: int, month: int) -> int:
    return year * 12 + month


def rotate_old_log_entries(
    log_path: Path = DEFAULT_LOG_PATH,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    keep_recent_months: int = 2,
    reference_date: date | None = None,
) -> list[Path]:
    """将早于保留窗口的 ``## [YYYY-MM-DD]`` 条目按季度归档到 ``docs/plans/archive/log-YYYY-QN.md``。

    保留顶部说明与最近 ``keep_recent_months`` 个自然月的记录，缩短主 ``log.md`` 体积。
    """
    if not log_path.is_file() or keep_recent_months < 1:
        return []
    text = log_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
    if len(blocks) <= 1:
        return []

    preamble = "" if blocks[0].startswith("## [") else blocks[0]
    entries = blocks[1:] if preamble else blocks

    ref = reference_date or date.today()
    cutoff_idx = _month_index(ref.year, ref.month) - (keep_recent_months - 1)

    kept: list[str] = []
    archived_by_quarter: dict[tuple[int, int], list[str]] = {}
    for entry in entries:
        match = _LOG_DATE_HEADING_RE.match(entry)
        if not match:
            kept.append(entry)
            continue
        year, month = int(match.group(1)), int(match.group(2))
        if _month_index(year, month) >= cutoff_idx:
            kept.append(entry)
        else:
            quarter = (month - 1) // 3 + 1
            archived_by_quarter.setdefault((year, quarter), []).append(entry)

    if not archived_by_quarter:
        return []

    archive_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (year, quarter), group in sorted(archived_by_quarter.items()):
        dest = archive_dir / f"log-{year}-Q{quarter}.md"
        header = f"# 知识库维护日志归档（{year} Q{quarter}）\n\n"
        new_body = "".join(b.strip("\n") + "\n\n" for b in group)
        if dest.is_file():
            existing = dest.read_text(encoding="utf-8")
            # 插在现有条目顶部，保留文件标题
            body_part = re.sub(r"^# [^\n]+\n+", "", existing, count=1)
            combined = header + new_body + body_part
        else:
            combined = header + new_body
        dest.write_text(combined.rstrip("\n") + "\n", encoding="utf-8")
        written.append(dest)

    rebuilt = preamble + "".join(kept)
    log_path.write_text(rebuilt.rstrip("\n") + "\n", encoding="utf-8")
    return written
