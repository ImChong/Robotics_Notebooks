import unittest
from pathlib import Path

from export_minimal import ROOT, parse_roadmap_stages

MOTION_CONTROL = ROOT / "roadmap" / "motion-control.md"
DEPTH_SAFE_CONTROL = ROOT / "roadmap" / "depth-safe-control.md"
DEPTH_RL_LOCOMOTION = ROOT / "roadmap" / "depth-rl-locomotion.md"


class RoadmapStagesExportTests(unittest.TestCase):
    def test_motion_control_parses_layer_stages(self) -> None:
        stages = parse_roadmap_stages(MOTION_CONTROL.read_text(encoding="utf-8"), MOTION_CONTROL)
        self.assertGreaterEqual(len(stages), 8)
        self.assertEqual(stages[0]["id"], "l0")
        self.assertIn("heading", stages[0])
        self.assertIn("数学与编程基础", stages[0]["heading"])

    def test_depth_safe_control_parses_stage_sections(self) -> None:
        stages = parse_roadmap_stages(DEPTH_SAFE_CONTROL.read_text(encoding="utf-8"), DEPTH_SAFE_CONTROL)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0]["id"], "stage-0")
        self.assertEqual(stages[-1]["id"], "stage-3")
        self.assertGreater(len(stages[1].get("related_items", [])), 0)

    def test_depth_rl_locomotion_parses_six_stages(self) -> None:
        stages = parse_roadmap_stages(DEPTH_RL_LOCOMOTION.read_text(encoding="utf-8"), DEPTH_RL_LOCOMOTION)
        self.assertEqual(len(stages), 6)
        self.assertEqual(stages[0]["id"], "stage-0")
        self.assertEqual(stages[-1]["id"], "stage-5")


if __name__ == "__main__":
    unittest.main()
