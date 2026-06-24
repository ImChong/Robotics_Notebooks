"""研究机构命名：中文（English） 格式校验（读 schema/institutions.json）。"""

from __future__ import annotations

import json
import unittest
from typing import Any

import generate_link_graph as glg

INSTITUTIONS_PATH = glg.INSTITUTIONS_REGISTRY_PATH


def _load_registry() -> dict[str, dict[str, Any]]:
    data = json.loads(INSTITUTIONS_PATH.read_text(encoding="utf-8"))
    registry = data.get("registry")
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"{INSTITUTIONS_PATH.name} 缺少非空 registry 对象")
    return registry


class InstitutionLabelPatternTest(unittest.TestCase):
    def test_valid_institution_labels(self) -> None:
        valid = [
            "英伟达（NVIDIA）",
            "清华大学（Tsinghua）",
            "地平线（Horizon Robotics）",
            "开源软件基金会（Linux Foundation）",
            "子弹物理仿真引擎（PyBullet / Bullet Physics）",
        ]
        for label in valid:
            with self.subTest(label=label):
                self.assertIsNotNone(glg.INSTITUTION_LABEL_RE.fullmatch(label))

    def test_invalid_institution_labels(self) -> None:
        invalid = [
            "NVIDIA",
            "Google DeepMind",
            "商汤科技 SenseNova",
            "Open Dynamic Robot Initiative（ODRI）",
            "香港科技大学广州校区",
        ]
        for label in invalid:
            with self.subTest(label=label):
                self.assertIsNone(glg.INSTITUTION_LABEL_RE.fullmatch(label))

    def test_registry_labels_conform_to_pattern(self) -> None:
        """schema/institutions.json 中全部 label 应符合 中文（English）。"""
        for inst_id, meta in _load_registry().items():
            label = str(meta.get("label", inst_id))
            with self.subTest(inst_id=inst_id, label=label):
                self.assertIsNotNone(
                    glg.INSTITUTION_LABEL_RE.fullmatch(label),
                    f"institution {inst_id!r} label={label!r}",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
