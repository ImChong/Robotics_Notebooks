"""更新记录热力图：固定 53 周滑动窗口（右缘锚定 UTC 今天）回归测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = ROOT / "docs" / "main.js"

# CI 不生成 gitignore 的 wiki-activity.json；用合成日数据验证窗口逻辑。
_SAMPLE_DAYS = [
    {"date": "2026-04-24", "count": 3, "added_count": 1, "maintained_count": 2},
    {"date": "2026-06-01", "count": 5, "added_count": 5},
    {"date": "2026-07-07", "count": 2, "maintained_count": 2},
]

_NODE_SNIPPET = r"""
const fs = require('fs');
const code = fs.readFileSync(process.argv[1], 'utf8');
const days = JSON.parse(process.argv[2]);
const mode = process.argv[3] || 'total';
const start = code.indexOf('var HOME_HEATMAP_DAY_MS');
const end = code.indexOf('function renderLatestWikiNode');
const vm = require('vm');
const sandbox = {};
vm.runInNewContext(code.slice(start, end), sandbox);
const html = sandbox.buildHomeWikiHeatmapHtml(
  days,
  mode === 'added' ? { addedOnly: true } : { addedOnly: false }
);
const gridMatch = html.match(/home-wiki-heatmap-grid[^>]*>([\s\S]*?)<\/div><\/div><\/div><\/div>/);
const gridHtml = gridMatch ? gridMatch[1] : '';
const gridCells = (gridHtml.match(/home-wiki-heatmap-cell/g) || []).length;
const weekAttr = html.match(/data-week-count="(\d+)"/);
const clickable = [...gridHtml.matchAll(/data-date="([^"]+)" data-count="(\d+)"/g)].map((m) => ({
  date: m[1],
  count: Number(m[2]),
}));
const modeAttr = (html.match(/data-count-mode="([^"]+)"/) || [])[1] || null;
const tipSample = (html.match(/title="([^"]+)"/) || [])[1] || '';
const todayMs = sandbox.homeHeatmapTodayUtcMs();
const bounds = sandbox.homeHeatmapWindowBounds(todayMs);
const weekMs = [];
for (let w = bounds.startMs; w <= bounds.endMs; w += sandbox.HOME_HEATMAP_DAY_MS * 7) weekMs.push(w);
console.log(JSON.stringify({
  gridCells,
  weekAttr: weekAttr ? Number(weekAttr[1]) : null,
  weeks: weekMs.length,
  today: new Date(todayMs).toISOString().slice(0, 10),
  start: new Date(bounds.startMs).toISOString().slice(0, 10),
  end: new Date(bounds.endMs).toISOString().slice(0, 10),
  modeAttr,
  tipSample,
  clickable,
}));
"""


def _run_node(
    days: list[dict[str, object]] | None = None,
    mode: str = "total",
) -> dict:
    payload = json.dumps(days if days is not None else _SAMPLE_DAYS)
    proc = subprocess.run(
        ["node", "-e", _NODE_SNIPPET, str(MAIN_JS), payload, mode],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def test_home_heatmap_fixed_53_week_window() -> None:
    out = _run_node()
    assert out["weeks"] == 53
    assert out["weekAttr"] == 53
    assert out["gridCells"] == 53 * 7


def test_home_heatmap_window_ends_on_today_week() -> None:
    out = _run_node()
    assert out["today"] <= out["end"]


def test_home_heatmap_window_not_data_span() -> None:
    """短跨度合成数据也不应退化为 min→max 全量列数（旧版约 12 周）。"""
    out = _run_node(_SAMPLE_DAYS)
    assert out["weeks"] == 53
    assert out["gridCells"] == 53 * 7


def test_home_heatmap_added_only_uses_added_count() -> None:
    out = _run_node(_SAMPLE_DAYS, mode="added")
    assert out["modeAttr"] == "added"
    by_date = {c["date"]: c["count"] for c in out["clickable"]}
    assert by_date.get("2026-04-24") == 1
    assert by_date.get("2026-06-01") == 5
    assert "2026-07-07" not in by_date  # 仅维护，默认不点亮
    assert "新增" in out["tipSample"]


def test_home_heatmap_total_uses_count() -> None:
    out = _run_node(_SAMPLE_DAYS, mode="total")
    assert out["modeAttr"] == "total"
    by_date = {c["date"]: c["count"] for c in out["clickable"]}
    assert by_date.get("2026-04-24") == 3
    assert by_date.get("2026-06-01") == 5
    assert by_date.get("2026-07-07") == 2
    assert "新增" not in out["tipSample"]
