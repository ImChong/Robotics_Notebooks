"""Pytest 入口：将 scripts/ 加入 import 路径与 CI 一致。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
