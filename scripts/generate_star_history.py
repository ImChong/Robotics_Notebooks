#!/usr/bin/env python3
"""generate_star_history.py — 部署时拉取 GitHub 星标时间线，供 docs/star-history.html 使用。

浏览器端直连 GitHub API 未登录限额仅 60 次/小时/IP，移动网络共享出口 IP 极易 403；
改为 pages.yml 构建时用 GITHUB_TOKEN 拉一次，写入 docs/exports/star-history.json
（只保留每个星标的日期，不含用户信息）。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO = "ImChong/Robotics_Notebooks"
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "exports" / "star-history.json"
PER_PAGE = 100


def fetch_page(page: int) -> list[dict]:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/stargazers?per_page={PER_PAGE}&page={page}",
        headers={
            "Accept": "application/vnd.github.star+json",
            "User-Agent": "Robotics-Notebooks-star-history",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        # 打出限额与响应体，便于在 Actions 日志里定位 403 原因
        print(
            f"❌ GitHub API {e.code} (page={page}, token={'yes' if token else 'no'}, "
            f"ratelimit-remaining={e.headers.get('X-RateLimit-Remaining')}, "
            f"ratelimit-resource={e.headers.get('X-RateLimit-Resource')}): "
            f"{e.read().decode('utf-8', 'replace')[:500]}",
            file=sys.stderr,
        )
        raise


def main() -> None:
    dates: list[str] = []
    page = 1
    while True:
        chunk = fetch_page(page)
        dates.extend(s["starred_at"][:10] for s in chunk)
        if len(chunk) < PER_PAGE:
            break
        page += 1
    dates.sort()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {"generated_at": date.today().isoformat(), "dates": dates}, separators=(",", ":")
        ),
        encoding="utf-8",
    )
    print(f"✅ star-history.json: {len(dates)} stars → {OUT_PATH}")


if __name__ == "__main__":
    main()
