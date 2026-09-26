#!/usr/bin/env python3
"""generate_star_history.py — 每日记录一次仓库星标总数，供 docs/star-history.html 使用。

由 weekly-lint.yml（每日定时）调用并提交 docs/exports/star-history.json。

带星标时间的 stargazers 接口必须登录（未登录 401、GITHUB_TOKEN 403），
故改为读取无需登录的仓库 stargazers_count，按日追加快照 [日期, 总数]。
2026-09-26 之前的历史由当日一次性导出的星标日期换算而来。
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "ImChong/Robotics_Notebooks"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "exports" / "star-history.json"


def fetch_star_count() -> int:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Robotics-Notebooks-star-history",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(json.load(resp)["stargazers_count"])


def main() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    counts = [c for c in data["counts"] if c[0] != today]
    counts.append([today, fetch_star_count()])
    data["counts"] = sorted(counts)
    OUT_PATH.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"✅ star-history.json: {today} → {data['counts'][-1][1]} stars")


if __name__ == "__main__":
    main()
