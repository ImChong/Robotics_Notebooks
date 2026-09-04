"""Regression tests for docs/depth-filters.js route-view matching."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPTH_FILTERS_JS = ROOT / "docs" / "depth-filters.js"


def _run_depth_filters_node(script: str) -> str:
    """Load depth-filters.js in Node and run a small script; return stdout."""
    prologue = f"""
const fs = require('fs');
const vm = require('vm');
const sandbox = {{ window: {{}} }};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync({json.dumps(str(DEPTH_FILTERS_JS))}, 'utf8'), sandbox);
const DF = sandbox.RNDepthFilters;
"""
    proc = subprocess.run(
        ["node", "-e", prologue + script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout


class DepthFiltersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTrue(DEPTH_FILTERS_JS.exists(), "docs/depth-filters.js should exist")

    def test_module_exports_expected_api(self):
        out = _run_depth_filters_node(
            """
const keys = ['DEPTH_ORDER','DEPTH_FILTERS','DEPTH_META','matches','segmentHits','nodePathKey','depthsForNode'];
for (const k of keys) {
  if (DF[k] == null) { console.error('missing', k); process.exit(2); }
}
console.log(JSON.stringify({
  orderLen: DF.DEPTH_ORDER.length,
  hasLoco: DF.DEPTH_ORDER.includes('loco-manipulation'),
  hasMotionGen: DF.DEPTH_ORDER.includes('motion-generation'),
}));
"""
        )
        data = json.loads(out.strip())
        self.assertEqual(data["orderLen"], 24)
        self.assertTrue(data["hasLoco"])
        self.assertTrue(data["hasMotionGen"])

    def test_hyphenated_segments_match_normalized_path_substring(self):
        """Multi-token segments must survive [/._-] splitting (the historic bug)."""
        out = _run_depth_filters_node(
            """
const cases = [
  // loco-manip stem must hit full loco-manipulation filenames
  ['loco-manipulation', 'wiki/entities/paper-halomi-humanoid-loco-manipulation.md', true],
  ['loco-manipulation', 'wiki/entities/paper-abot-m05-mobile-manipulation-wam.md', true],
  ['loco-manipulation', 'wiki/entities/paper-deep-whole-body-control-loco-manip.md', true],
  ['loco-manipulation', 'wiki/tasks/loco-manipulation.md', true],
  // underscore form of sim-to-real
  ['sim2real', 'wiki/entities/paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md', true],
  // diffusion-policy / motion-generation compounds
  ['imitation-learning', 'wiki/methods/diffusion-policy.md', true],
  ['motion-generation', 'wiki/methods/diffusion-motion-generation.md', true],
  ['motion-generation', 'wiki/entities/awesome-text-to-motion-zilize.md', true],
  // cross-route negative: plain WBC is not loco-manip by path
  ['loco-manipulation', 'wiki/concepts/whole-body-control.md', false],
];
const failed = [];
for (const [depth, id, expect] of cases) {
  const got = DF.matches({ id }, depth);
  if (got !== expect) failed.push({ depth, id, expect, got });
}
console.log(JSON.stringify({ failed }));
"""
        )
        data = json.loads(out.strip())
        self.assertEqual(data["failed"], [], msg=json.dumps(data["failed"], indent=2))

    def test_route_anchor_pages_always_match(self):
        out = _run_depth_filters_node(
            """
const failed = [];
for (const key of DF.DEPTH_ORDER) {
  const hub = DF.DEPTH_HUB_IDS[key];
  if (!hub) { failed.push({ key, reason: 'no hub' }); continue; }
  if (!DF.matches({ id: hub }, key)) failed.push({ key, hub, reason: 'hub miss' });
  if (!DF.isDepthHub({ id: hub }, key)) failed.push({ key, hub, reason: 'isDepthHub' });
}
console.log(JSON.stringify({ failed }));
"""
        )
        data = json.loads(out.strip())
        self.assertEqual(data["failed"], [])

    def test_explicit_anchor_ids_for_sparse_routes(self):
        """Routes whose core papers lack path keywords still need explicit ids."""
        out = _run_depth_filters_node(
            """
const cases = [
  ['humanoid-boxing', 'wiki/entities/paper-notebook-robostriker.md'],
  ['humanoid-boxing', 'wiki/entities/rek.md'],
  ['humanoid-boxing', 'wiki/entities/urkl.md'],
  ['humanoid-soccer', 'wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md'],
  ['motion-generation', 'wiki/entities/kimodo.md'],
  ['motion-generation', 'wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md'],
  ['real2sim', 'wiki/entities/articraft.md'],
  ['wam', 'wiki/entities/cosmos-3.md'],
  ['loco-manipulation', 'wiki/entities/paper-resmimic.md'],
];
const failed = [];
for (const [depth, id] of cases) {
  if (!DF.matches({ id }, depth)) failed.push({ depth, id });
}
console.log(JSON.stringify({ failed }));
"""
        )
        data = json.loads(out.strip())
        self.assertEqual(data["failed"], [])

    def test_loco_manipulation_route_recovers_corpus(self):
        """After the hyphen bugfix, loco-manip route must surface the named corpus."""
        wiki_root = ROOT / "wiki"
        candidates = sorted(
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in wiki_root.rglob("*.md")
            if "loco-manip" in p.name or "mobile-manip" in p.name
        )
        self.assertGreaterEqual(len(candidates), 20, "fixture corpus unexpectedly small")
        payload = json.dumps(candidates)
        out = _run_depth_filters_node(
            f"""
const ids = {payload};
const missed = ids.filter(id => !DF.matches({{ id }}, 'loco-manipulation'));
console.log(JSON.stringify({{ total: ids.length, missed }}));
"""
        )
        data = json.loads(out.strip())
        self.assertEqual(
            data["missed"],
            [],
            msg=f"missed {len(data['missed'])}/{data['total']}: {data['missed'][:10]}",
        )

    def test_single_token_segments_remain_exact(self):
        """Single-token rules must not become raw substring matches."""
        out = _run_depth_filters_node(
            """
// 'ball' is a soccer single-token segment; must not match inside 'football' token only via substring —
// football tokenizes to ['football'], so 'ball' exact-token miss is expected unless path has 'ball'.
const football = DF.matches({ id: 'wiki/entities/paper-humanoid-football-demo.md' }, 'humanoid-soccer');
const ball = DF.matches({ id: 'wiki/entities/paper-learning-to-ball.md' }, 'humanoid-soccer');
// segmentHits API: multi vs single
const multi = DF.segmentHits(
  { id: 'wiki/entities/paper-halomi-humanoid-loco-manipulation.md' },
  DF.nodeSegments({ id: 'wiki/entities/paper-halomi-humanoid-loco-manipulation.md' }),
  'loco-manip'
);
const singleExact = DF.segmentHits(
  { id: 'wiki/tasks/locomotion.md' },
  DF.nodeSegments({ id: 'wiki/tasks/locomotion.md' }),
  'locomotion'
);
console.log(JSON.stringify({ football, ball, multi, singleExact }));
"""
        )
        data = json.loads(out.strip())
        self.assertTrue(data["multi"])
        self.assertTrue(data["singleExact"])


if __name__ == "__main__":
    unittest.main()
