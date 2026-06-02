import unittest
from pathlib import Path

from export_minimal import clean_summary, extract_summary
from search_indexing import parse_frontmatter, strip_frontmatter

ROOT = Path(__file__).resolve().parents[1]
SAFE_RL = ROOT / "wiki" / "concepts" / "safe-real-world-rl-fine-tuning.md"


class DetailSummaryExportTests(unittest.TestCase):
    def test_clean_summary_strips_markdown_links(self) -> None:
        self.assertEqual(
            clean_summary("关心 [Sim2Real](./sim2real.md) 链路"),
            "关心 Sim2Real 链路。",
        )

    def test_extract_summary_prefers_frontmatter(self) -> None:
        text = SAFE_RL.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        body = strip_frontmatter(text)
        summary = extract_summary(body, fm)
        self.assertIn("真机安全 RL 微调", summary)
        self.assertNotIn("[Sim2Real]", summary)
        self.assertNotIn("](./", summary)

    def test_intro_paragraph_with_colon_is_not_labeled_summary(self) -> None:
        sample = (
            "# Title\n\n"
            "**真机安全 RL 微调** 关心 [Sim2Real](./sim2real.md) 链路的**最后一段**：正文。\n\n"
            "## 一句话定义\n\n"
            "在真机上抠最后几成性能。\n"
        )
        summary = extract_summary(sample, {})
        self.assertIn("在真机上抠最后几成性能", summary)


if __name__ == "__main__":
    unittest.main()
