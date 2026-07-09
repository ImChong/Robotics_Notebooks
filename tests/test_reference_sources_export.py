import unittest
from pathlib import Path

from export_minimal import collect_reference_sources

ROOT = Path(__file__).resolve().parents[1]
SIM2REAL = ROOT / "wiki" / "concepts" / "sim2real.md"


class ReferenceSourcesExportTests(unittest.TestCase):
    def test_collect_reference_sources_from_section(self) -> None:
        text = SIM2REAL.read_text(encoding="utf-8")
        sources = collect_reference_sources(text, SIM2REAL)
        self.assertGreaterEqual(len(sources), 10)
        labels = [entry["label"] for entry in sources]
        self.assertTrue(any("KungFuAthleteBot" in label for label in labels))
        self.assertFalse(any(label.startswith("Tobin") for label in labels))
        detail_ids = [entry.get("detail_id", "") for entry in sources]
        self.assertIn("wiki-comparisons-sim2real-approaches", detail_ids)
        github_urls = [entry.get("url", "") for entry in sources]
        self.assertTrue(
            any(
                url.startswith("https://github.com/ImChong/Robotics_Notebooks/blob/main/sources/")
                for url in github_urls
            )
        )

    def test_collect_reference_sources_fallback_to_external_links(self) -> None:
        sample = "# Title\n\n## 核心内容\n\n正文含 https://example.com/paper 链接。\n"
        path = ROOT / "wiki" / "concepts" / "sample.md"
        sources = collect_reference_sources(sample, path)
        self.assertEqual(
            sources, [{"label": "https://example.com/paper", "url": "https://example.com/paper"}]
        )

    def test_collect_reference_sources_excludes_plain_text_entry(self) -> None:
        sample = (
            "## 参考来源\n\n"
            "- Tobin et al. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*\n"
            "- [Paper notes](https://example.com/paper)\n"
        )
        path = ROOT / "wiki" / "concepts" / "sample.md"
        sources = collect_reference_sources(sample, path)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.com/paper")
        labels = [entry["label"] for entry in sources]
        self.assertFalse(any("Tobin" in label for label in labels))

    def test_clean_reference_label_strips_angle_bracket_autolinks(self) -> None:
        from export_minimal import clean_reference_label, parse_reference_line

        path = ROOT / "wiki" / "concepts" / "sample.md"
        samples = [
            (
                "- FastStair 论文 HTML：<https://arxiv.org/html/2601.10365v1>",
                "FastStair 论文 HTML",
            ),
            (
                "- [wechat.md](../../sources/blogs/wechat.md) — <https://mp.weixin.qq.com/s/example>",
                "wechat.md",
            ),
            (
                "- 论文 PDF：<https://arxiv.org/pdf/2602.08602>",
                "论文 PDF",
            ),
            (
                "- sources/papers/barkour.md — Barkour：>1m/s 敏捷动作",
                "sources/papers/barkour.md — Barkour：>1m/s 敏捷动作",
            ),
        ]
        for line, expected in samples:
            entries = parse_reference_line(line, path)
            self.assertEqual(entries[0]["label"], expected, msg=line)
            self.assertNotIn("<", entries[0]["label"])
        self.assertEqual(clean_reference_label("BOM &lt;$400"), "BOM <$400")


if __name__ == "__main__":
    unittest.main()
