"""首页「最新知识节点」紧凑列表：新增不足时从 wiki-activity 自新到旧回填。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = ROOT / "docs" / "main.js"

_NODE_SNIPPET = r"""
const fs = require('fs');
const code = fs.readFileSync(process.argv[1], 'utf8');
const payload = JSON.parse(process.argv[2]);
const start = code.indexOf('function collectHomeCompactAddedNodes');
const end = code.indexOf('function renderUpdatesItemSuffix');
if (start < 0 || end < 0 || end <= start) {
  console.error('collectHomeCompactAddedNodes not found');
  process.exit(2);
}
const vm = require('vm');
const sandbox = {};
vm.runInNewContext(code.slice(start, end), sandbox);
const out = sandbox.collectHomeCompactAddedNodes(
  payload.items,
  payload.wikiActivity,
  payload.maxItems
);
console.log(JSON.stringify({
  ids: out.map((n) => n.detail_id),
  dates: out.map((n) => n.recency || ''),
  actions: out.map((n) => n.action),
}));
"""

_SAMPLE = {
    "items": [
        {
            "detail_id": "new-a",
            "label": "A",
            "action": "added",
            "recency": "2026-08-12",
        },
        {
            "detail_id": "maint-x",
            "label": "X",
            "action": "maintained",
            "recency": "2026-08-12",
        },
        {
            "detail_id": "new-b",
            "label": "B",
            "action": "added",
            "recency": "2026-08-12",
        },
    ],
    "wikiActivity": {
        "days": [
            {
                "date": "2026-04-24",
                "nodes": [
                    {"detail_id": "ancient", "label": "Ancient", "action": "added"},
                ],
            },
            {
                "date": "2026-08-11",
                "nodes": [
                    {"detail_id": "old-maint", "label": "M", "action": "maintained"},
                    {"detail_id": "recent-c", "label": "C", "action": "added"},
                    {"detail_id": "recent-d", "label": "D", "action": "added"},
                ],
            },
            {
                "date": "2026-08-12",
                "nodes": [
                    {"detail_id": "new-a", "label": "A", "action": "added"},
                    {"detail_id": "new-b", "label": "B", "action": "added"},
                    {"detail_id": "recent-e", "label": "E", "action": "added"},
                ],
            },
        ]
    },
    "maxItems": 5,
}


def _run(payload: dict) -> dict:
    proc = subprocess.run(
        ["node", "-e", _NODE_SNIPPET, str(MAIN_JS), json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def test_backfill_prefers_recent_days_not_oldest() -> None:
    out = _run(_SAMPLE)
    assert out["ids"] == ["new-a", "new-b", "recent-e", "recent-c", "recent-d"]
    assert "ancient" not in out["ids"]
    assert out["dates"] == [
        "2026-08-12",
        "2026-08-12",
        "2026-08-12",
        "2026-08-11",
        "2026-08-11",
    ]
    assert out["actions"] == ["added"] * 5


def test_backfill_sets_recency_from_activity_day() -> None:
    out = _run(
        {
            "items": [],
            "wikiActivity": {
                "days": [
                    {
                        "date": "2026-08-10",
                        "nodes": [
                            {"detail_id": "only", "label": "Only", "action": "added"},
                        ],
                    }
                ]
            },
            "maxItems": 3,
        }
    )
    assert out["ids"] == ["only"]
    assert out["dates"] == ["2026-08-10"]


def test_no_activity_keeps_latest_added_only() -> None:
    out = _run(
        {
            "items": _SAMPLE["items"],
            "wikiActivity": None,
            "maxItems": 5,
        }
    )
    assert out["ids"] == ["new-a", "new-b"]
    assert out["dates"] == ["2026-08-12", "2026-08-12"]
