"""共享的 wiki 页面扫描缓存。

`wiki/*.md` 在一次脚本运行内通常被多处重复 `rglob`，对 1000+ 文件的目录树
这会带来可观的重复 I/O。本模块用进程级缓存把扫描摊销到一次。

⚠️ 缓存有效期为整个进程生命周期，**仅在 wiki 文件集合不变时可复用**。
只读导出脚本（export_minimal、generate_link_graph 等）可放心使用；
会在运行中新增/删除 wiki .md 的脚本不要用本缓存，否则会读到过期结果。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

WIKI_DIR = Path(__file__).resolve().parents[2] / "wiki"


@lru_cache(maxsize=1)
def wiki_stem_to_path() -> dict[str, Path]:
    """返回 {文件名(stem) -> 路径} 索引，用于解析 [[stem]] wikilink。

    返回的 dict 被缓存复用，调用方只读、不要原地修改。
    """
    return {p.stem: p for p in WIKI_DIR.rglob("*.md")}
