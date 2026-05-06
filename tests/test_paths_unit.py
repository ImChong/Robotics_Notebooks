import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.paths import slugify, path_to_id

class TestPathsUnit(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello World!"), "hello-world")
        self.assertEqual(slugify("Hello World"), "hello-world")
        self.assertEqual(slugify("a--b"), "a-b")
        self.assertEqual(slugify("---a---b---"), "a-b")

    def test_path_to_id(self):
        root = Path("/fake/root")

        self.assertEqual(path_to_id(root / "wiki/entities/robot.md", root), "entity-robot")
        self.assertEqual(path_to_id(root / "wiki/methods/mpc.md", root), "wiki-methods-mpc")
        self.assertEqual(path_to_id(root / "roadmap/motion-control.md", root), "roadmap-motion-control")
        self.assertEqual(path_to_id(root / "references/papers/rl.md", root), "reference-papers-rl")

        self.assertEqual(path_to_id(root / "tech-map/modules/control/mpc.md", root), "tech-node-control-mpc")
        self.assertEqual(path_to_id(root / "tech-map/research-directions/humanoid.md", root), "tech-node-research-humanoid")
        self.assertEqual(path_to_id(root / "tech-map/overview.md", root), "tech-node-overview")

        self.assertEqual(path_to_id(root / "other/folder/file.md", root), "other-folder-file")

if __name__ == "__main__":
    unittest.main()
