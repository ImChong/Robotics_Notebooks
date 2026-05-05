import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_indexing import parse_frontmatter
from search_wiki import _filter_doc, _find_matched_lines

class TestSearchIndexingUnit(unittest.TestCase):
    def test_parse_frontmatter_empty(self):
        self.assertEqual(parse_frontmatter(""), {})

    def test_parse_frontmatter_no_delimiter(self):
        self.assertEqual(parse_frontmatter("title: Hello\nNo frontmatter here"), {})

    def test_parse_frontmatter_incomplete(self):
        self.assertEqual(parse_frontmatter("---\ntitle: Hello\n"), {})

    def test_parse_frontmatter_basic(self):
        content = "---\ntitle: Test\nauthor: Jules\n---\nBody content"
        expected = {"title": "Test", "author": "Jules"}
        self.assertEqual(parse_frontmatter(content), expected)

    def test_parse_frontmatter_list(self):
        content = "---\ntags: [robotics, control, ai]\n---\n"
        expected = {"tags": ["robotics", "control", "ai"]}
        self.assertEqual(parse_frontmatter(content), expected)

    def test_parse_frontmatter_list_with_quotes(self):
        content = "---\ntags: [\"robotics\", 'control', ai]\n---\n"
        expected = {"tags": ["robotics", "control", "ai"]}
        self.assertEqual(parse_frontmatter(content), expected)

    def test_parse_frontmatter_quoted_value(self):
        content = "---\ntitle: \"Quoted Title\"\ndescription: 'Single Quoted'\n---\n"
        expected = {"title": "Quoted Title", "description": "Single Quoted"}
        self.assertEqual(parse_frontmatter(content), expected)

    def test_parse_frontmatter_comments_and_empty_lines(self):
        content = """---
title: Test
# This is a comment

key: value
---
"""
        expected = {"title": "Test", "key": "value"}
        self.assertEqual(parse_frontmatter(content), expected)



class TestSearchWikiHelpers(unittest.TestCase):
    def test_filter_doc_type(self):
        doc = {"page_type": "concept", "tags": ["rl"]}
        self.assertTrue(_filter_doc(doc, type_filter="concept", tag_filters=None))
        self.assertFalse(_filter_doc(doc, type_filter="method", tag_filters=None))

    def test_filter_doc_tags(self):
        doc = {"page_type": "concept", "tags": ["rl", "control"]}
        self.assertTrue(_filter_doc(doc, type_filter=None, tag_filters=["rl"]))
        self.assertTrue(_filter_doc(doc, type_filter=None, tag_filters=["rl", "control"]))
        self.assertFalse(_filter_doc(doc, type_filter=None, tag_filters=["rl", "missing"]))

    def test_filter_doc_both(self):
        doc = {"page_type": "concept", "tags": ["rl"]}
        self.assertTrue(_filter_doc(doc, type_filter="concept", tag_filters=["rl"]))
        self.assertFalse(_filter_doc(doc, type_filter="method", tag_filters=["rl"]))
        self.assertFalse(_filter_doc(doc, type_filter="concept", tag_filters=["missing"]))

    def test_find_matched_lines_no_words(self):
        lines = ["hello world", "test line"]
        self.assertEqual(_find_matched_lines(lines, [], 1), [])

    def test_find_matched_lines_match(self):
        lines = ["line 1", "hello world", "line 3", "line 4"]
        matches = _find_matched_lines(lines, ["hello"], 1)
        # Expect [(line_num_1_based, [context_lines], match_offset_in_context)]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 2) # line index 1 is line 2
        self.assertEqual(matches[0][1], ["line 1", "hello world", "line 3"])
        self.assertEqual(matches[0][2], 1)

    def test_find_matched_lines_case_insensitive(self):
        lines = ["line 1", "HELLO WORLD", "line 3", "line 4"]
        matches = _find_matched_lines(lines, ["hello"], 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 2)
        self.assertEqual(matches[0][1], ["line 1", "HELLO WORLD", "line 3"])
        self.assertEqual(matches[0][2], 1)

if __name__ == "__main__":
    unittest.main()
