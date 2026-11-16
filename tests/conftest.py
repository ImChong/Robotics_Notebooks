"""Pytest 入口：将 scripts/ 加入 import 路径与 CI 一致，并缓存只读全图构建。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True, scope="session")
def _cache_default_build_graph_data() -> Any:
    """在同一 pytest 会话内复用默认仓库参数的 _build_graph_data()，返回深拷贝防串扰。"""
    import generate_link_graph as glg

    original = glg._build_graph_data
    default_dirs = tuple(glg.GRAPH_CONTENT_DIRS)
    default_root = glg.REPO_ROOT
    cached: tuple[list[dict[str, Any]], list[dict[str, str]]] | None = None

    def cached_build() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        nonlocal cached
        if glg.REPO_ROOT != default_root or tuple(glg.GRAPH_CONTENT_DIRS) != default_dirs:
            return original()
        if cached is None:
            cached = original()
        return copy.deepcopy(cached[0]), copy.deepcopy(cached[1])

    glg._build_graph_data = cached_build
    try:
        yield
    finally:
        glg._build_graph_data = original
