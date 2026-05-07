"""Unit tests for scripts/search_wiki_core scoring helpers (raises coverage on pure logic)."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_indexing import REPO_ROOT
from search_wiki_core import (
    collect_known_terms,
    compute_avgdl,
    compute_score,
    extract_related_links,
    levenshtein_distance,
    normalize_scores,
    suggest_terms,
)


class TestLevenshtein(unittest.TestCase):
    def test_equal_and_empty(self):
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(levenshtein_distance("", "xy"), 2)
        self.assertEqual(levenshtein_distance("a", ""), 1)

    def test_edit_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)


class TestNormalizeScores(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(normalize_scores([]), [])

    def test_uniform_positive(self):
        self.assertEqual(normalize_scores([3.0, 3.0, 3.0]), [1.0, 1.0, 1.0])

    def test_uniform_non_positive(self):
        self.assertEqual(normalize_scores([-1.0, -1.0]), [0.0, 0.0])

    def test_range(self):
        self.assertEqual(normalize_scores([0.0, 10.0]), [0.0, 1.0])


class TestComputeScore(unittest.TestCase):
    def test_no_query_tokens(self):
        self.assertEqual(
            compute_score({"a": 1}, [], title="t", avgdl=5.0, fm={}, page_type="concept"),
            0.0,
        )

    def test_title_boost(self):
        tc = {"mpc": 2}
        s_title = compute_score(
            tc, ["mpc"], title="MPC intro", avgdl=10.0, fm={}, page_type="method"
        )
        s_not = compute_score(tc, ["mpc"], title="other", avgdl=10.0, fm={}, page_type="method")
        self.assertGreater(s_title, s_not)

    def test_summary_boost(self):
        tc = {"foo": 1}
        fm = {"summary": "Contains foo term"}
        with_summary = compute_score(
            tc, ["foo"], title="none", avgdl=5.0, fm=fm, page_type="concept"
        )
        without = compute_score(tc, ["foo"], title="none", avgdl=5.0, fm={}, page_type="concept")
        self.assertGreater(with_summary, without)

    def test_page_type_multipliers(self):
        tc = {"x": 1}
        base = compute_score(tc, ["x"], title="", avgdl=5.0, fm={}, page_type="concept")
        q = compute_score(tc, ["x"], title="", avgdl=5.0, fm={}, page_type="query")
        c = compute_score(tc, ["x"], title="", avgdl=5.0, fm={}, page_type="comparison")
        self.assertLess(q, base)
        self.assertGreater(c, base)

    def test_recent_updated_boost(self):
        tc = {"z": 1}
        today = date.today().isoformat()
        recent = compute_score(
            tc, ["z"], title="", avgdl=3.0, fm={"updated": today}, page_type="concept"
        )
        old = compute_score(
            tc,
            ["z"],
            title="",
            avgdl=3.0,
            fm={"updated": "1990-01-01"},
            page_type="concept",
        )
        self.assertGreater(recent, old)


class TestComputeAvgdl(unittest.TestCase):
    def test_basic(self):
        docs = [
            {"token_counts": {"a": 3}},
            {"token_counts": {"b": 1}},
        ]
        self.assertAlmostEqual(compute_avgdl(docs), 2.0)


class TestCollectKnownTerms(unittest.TestCase):
    def test_merges_title_and_tags(self):
        docs = [
            {"title": "Hello", "tags": ["RL", "control"]},
            {"title": "hello", "tags": ["RL"]},
        ]
        terms = collect_known_terms(docs)
        self.assertEqual(terms["hello"], "Hello")
        self.assertEqual(terms["rl"], "RL")


class TestSuggestTerms(unittest.TestCase):
    def test_typo_within_distance(self):
        terms = {"locomotion": "locomotion"}
        out = suggest_terms("lokomotion", terms, top_k=3)
        self.assertTrue(any(t[0] == "locomotion" for t in out))

    def test_empty_query(self):
        self.assertEqual(suggest_terms("   ", {"a": "A"}, top_k=3), [])


class TestExtractRelatedLinks(unittest.TestCase):
    def test_resolves_relative_md(self):
        src = REPO_ROOT / "wiki" / "concepts" / "armature-modeling.md"
        content = "## 关联页面\n\n- [LIP ZMP](lip-zmp.md)\n"
        links = extract_related_links(content, src)
        self.assertTrue(any("lip-zmp.md" in entry for entry in links))


if __name__ == "__main__":
    unittest.main()
