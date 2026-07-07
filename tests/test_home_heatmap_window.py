"""更新记录热力图：固定 53 周滑动窗口（右缘锚定 UTC 今天）回归测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = ROOT / "docs" / "main.js"
WIKI_ACTIVITY = ROOT / "docs" / "exports" / "wiki-activity.json"

_NODE_SNIPPET = r"""
const fs = require('fs');
const code = fs.readFileSync(process.argv[1], 'utf8');
const start = code.indexOf('var HOME_HEATMAP_DAY_MS');
const end = code.indexOf('function renderLatestWikiNode');
const vm = require('vm');
const sandbox = {};
vm.runInNewContext(code.slice(start, end), sandbox);
const days = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).days;
const html = sandbox.buildHomeWikiHeatmapHtml(days);
const gridMatch = html.match(/home-wiki-heatmap-grid[^>]*>([\s\S]*?)<\/div><\/div><\/div><\/div>/);
const gridHtml = gridMatch ? gridMatch[1] : '';
const gridCells = (gridHtml.match(/home-wiki-heatmap-cell/g) || []).length;
const weekAttr = html.match(/data-week-count="(\d+)"/);
const todayMs = sandbox.homeHeatmapTodayUtcMs();
const bounds = sandbox.homeHeatmapWindowBounds(todayMs);
const weekMs = [];
for (let w = bounds.startMs; w <= bounds.endMs; w += sandbox.HOME_HEATMAP_DAY_MS * 7) weekMs.push(w);
console.log(JSON.stringify({
  gridCells,
  weekAttr: weekAttr ? Number(weekAttr[1]) : null,
  weeks: weekMs.length,
  start: new Date(bounds.startMs).toISOString().slice(0, 10),
  end: new Date(bounds.endMs).toISOString().slice(0, 10),
}));
"""


def _run_node() -> dict:
    if not WIKI_ACTIVITY.is_file():
        raise FileNotFoundError(
            "docs/exports/wiki-activity.json missing; run `make export graph` first"
        )
    proc = subprocess.run(
        ["node", "-e", _NODE_SNIPPET, str(MAIN_JS), str(WIKI_ACTIVITY)],
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
    proc = subprocess.run(
        ["node", "-e", "const n=new Date(); console.log(n.toISOString().slice(0,10));"],
        check=True,
        capture_output=True,
        text=True,
    )
    today = proc.stdout.strip()
    assert today <= out["end"]
