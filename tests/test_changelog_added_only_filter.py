"""更新记录页「默认仅新增 / 显示维护节点」过滤逻辑回归测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = ROOT / "docs" / "main.js"

_SAMPLE_GROUPS = [
    {
        "date": "2026-08-07",
        "items": [
            {"detail_id": "a", "action": "added"},
            {"detail_id": "b", "action": "maintained"},
        ],
        "totalCount": 2,
        "addedCount": 1,
        "maintainedCount": 1,
    },
    {
        "date": "2026-08-06",
        "items": [
            {"detail_id": "c", "action": "maintained"},
        ],
        "totalCount": 1,
        "addedCount": 0,
        "maintainedCount": 1,
    },
    {
        "date": "2026-08-05",
        "items": [
            {"detail_id": "d", "action": "added"},
            {"detail_id": "e", "action": "added"},
        ],
        "totalCount": 2,
        "addedCount": 2,
        "maintainedCount": 0,
    },
]

_NODE_SNIPPET = r"""
const fs = require('fs');
const code = fs.readFileSync(process.argv[1], 'utf8');
const groups = JSON.parse(process.argv[2]);
const start = code.indexOf('function renderLatestWikiNode');
const end = code.indexOf('function moduleHref');
const slice = code.slice(start, end);
// 抽出 renderLatestWikiNode 内部的过滤辅助函数，挂到 sandbox
const helpers = slice.match(
  /function filterMetasAddedOnly[\s\S]*?function formatDayMeta/
);
if (!helpers) {
  console.error('helpers not found');
  process.exit(2);
}
const vm = require('vm');
const sandbox = {};
vm.runInNewContext(
  helpers[0].replace(/function formatDayMeta[\s\S]*$/, ''),
  sandbox
);
const filtered = sandbox.filterTimelineGroupsAddedOnly(groups);
console.log(JSON.stringify({
  dayCount: filtered.length,
  dates: filtered.map((g) => g.date),
  counts: filtered.map((g) => g.items.length),
  actions: filtered.flatMap((g) => g.items.map((i) => i.action)),
  maintainedDropped: !filtered.some((g) => g.date === '2026-08-06'),
}));
"""


def _run_node(groups: list[dict] | None = None) -> dict:
    payload = json.dumps(groups if groups is not None else _SAMPLE_GROUPS)
    proc = subprocess.run(
        ["node", "-e", _NODE_SNIPPET, str(MAIN_JS), payload],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def test_filter_keeps_only_added_and_drops_empty_days() -> None:
    out = _run_node()
    assert out["dayCount"] == 2
    assert out["dates"] == ["2026-08-07", "2026-08-05"]
    assert out["counts"] == [1, 2]
    assert out["actions"] == ["added", "added", "added"]
    assert out["maintainedDropped"] is True


def test_filter_all_maintained_yields_empty() -> None:
    out = _run_node(
        [
            {
                "date": "2026-08-01",
                "items": [{"detail_id": "x", "action": "maintained"}],
                "totalCount": 1,
                "addedCount": 0,
                "maintainedCount": 1,
            }
        ]
    )
    assert out["dayCount"] == 0
    assert out["dates"] == []
