import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_search_index import generate_search_index  # noqa: E402
from utils.community_labels import (  # noqa: E402
    community_search_aliases,
    community_search_aliases_for_path,
    community_short_label,
)


class CommunitySearchAliasTests(unittest.TestCase):
    def test_community_short_label_strips_suffix_and_english(self) -> None:
        full = "人形论文深读笔记（Humanoid Paper Notebooks） 社区"
        self.assertEqual(community_short_label(full), "人形论文深读笔记")

    def test_community_search_aliases_for_paper_notebooks_hub(self) -> None:
        aliases = community_search_aliases_for_path(
            "wiki/overview/humanoid-paper-notebooks-index.md"
        )
        self.assertIn("人形论文深读笔记", aliases)
        self.assertIn("Humanoid Paper Notebooks", aliases)

    def test_community_search_aliases_from_base_name(self) -> None:
        aliases = community_search_aliases("视觉-语言-动作（Vision-Language-Action, VLA）")
        self.assertEqual(aliases[0], "视觉-语言-动作")
        self.assertIn("Vision-Language-Action, VLA", aliases)

    def test_search_index_includes_aliases_for_paper_notebooks_hub(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "search-index.json"
            payload = generate_search_index(output_path)

        doc = next(
            d
            for d in payload["docs"]
            if d["path"] == "wiki/overview/humanoid-paper-notebooks-index.md"
        )
        self.assertIn("人形论文深读笔记", doc.get("search_aliases", []))
        self.assertGreater(doc["tokens"].get("人形论文深读笔记", 0), 0)


if __name__ == "__main__":
    unittest.main()
