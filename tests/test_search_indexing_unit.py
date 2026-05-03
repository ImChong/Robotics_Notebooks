import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_indexing import parse_frontmatter

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

if __name__ == "__main__":
    unittest.main()
