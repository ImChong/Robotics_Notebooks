from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_search_index import generate_search_index  # noqa: E402


class SearchIndexTests(unittest.TestCase):
    def test_search_index_includes_tech_map_nodes(self):
        payload = generate_search_index(ROOT / "docs" / "search-index.json")
        docs = payload["docs"]
        tech_docs = [doc for doc in docs if doc["path"].startswith("tech-map/")]

        self.assertTrue(tech_docs)
        self.assertIn("tech-node-overview", {doc["id"] for doc in tech_docs})
        self.assertTrue(any(doc["tokens"].get("tech-map", 0) > 0 for doc in tech_docs))


if __name__ == "__main__":
    unittest.main()
