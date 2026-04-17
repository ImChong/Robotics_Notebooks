#!/usr/bin/env python3
"""update_badge.py — 从 lint 输出中读取覆盖率数值，更新 README.md 中的 Sources Coverage badge。"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"

result = subprocess.run(
    ["python3", "scripts/lint_wiki.py"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
m = re.search(r"(\d+)/\d+ \((\d+)%\)", result.stdout)
if not m:
    print("Coverage not found in lint output:", result.stdout[-200:], file=sys.stderr)
    sys.exit(1)

pct = int(m.group(2))
color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
badge = f"[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-{pct}%25-{color})](docs/tech-stack-next-phase-checklist-v9.md)"

content = README.read_text(encoding="utf-8")
new_content = re.sub(
    r"\[!\[Sources Coverage\]\([^)]+\)\]\([^)]+\)",
    badge,
    content,
)
README.write_text(new_content, encoding="utf-8")
print(f"Badge updated: {pct}% ({color})")
