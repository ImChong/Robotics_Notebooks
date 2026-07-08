import unittest
from pathlib import Path

from export_minimal import (
    ROADMAP_HERO_LABEL,
    extract_body_markdown,
    extract_labeled_bullets,
    extract_roadmap_hero,
    extract_summary,
    parse_roadmap_stages,
    strip_labeled_section,
)

ROOT = Path(__file__).resolve().parents[1]
MOTION_CONTROL = ROOT / "roadmap" / "motion-control.md"


class RoadmapHeroExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = MOTION_CONTROL.read_text(encoding="utf-8")

    def test_motion_control_has_hero_section(self) -> None:
        items = extract_labeled_bullets(self.text, ROADMAP_HERO_LABEL)
        self.assertGreaterEqual(len(items), 3)
        self.assertIn("为谁", items[0])

    def test_extract_summary_uses_abstract_not_hero(self) -> None:
        summary = extract_summary(self.text)
        self.assertIn("L7", summary)
        self.assertIn("L−1", summary)
        self.assertNotIn("为谁", summary)

    def test_body_strips_hero_keeps_abstract(self) -> None:
        body = extract_body_markdown(self.text)
        self.assertNotIn("首屏导读", body)
        self.assertIn("**摘要**", body)
        self.assertIn("一条主线", body)

    def test_hero_short_for_meta(self) -> None:
        items, short = extract_roadmap_hero(self.text)
        self.assertEqual(len(items), 3)
        self.assertTrue(short)
        self.assertLessEqual(len(short), 130)

    def test_strip_labeled_section(self) -> None:
        sample = "# T\n\n**首屏导读**：\n\n- a\n\n**摘要**：\n\n- b\n"
        out = strip_labeled_section(sample.split("\n", 1)[1], ROADMAP_HERO_LABEL)
        self.assertNotIn("首屏导读", out)
        self.assertIn("**摘要**", out)

    def test_parse_roadmap_stages_includes_l_minus_one(self) -> None:
        stages = parse_roadmap_stages(self.text, MOTION_CONTROL)
        ids = [stage["id"] for stage in stages]
        self.assertEqual(ids[0], "l-1")
        self.assertIn("l0", ids)
        self.assertIn("l7", ids)
        self.assertGreaterEqual(len(ids), 9)


if __name__ == "__main__":
    unittest.main()
