# JEPLO（arXiv:2609.15770）

> 来源归档（paper）

- **标题：** JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15770>
- **PDF：** <https://arxiv.org/pdf/2609.15770>
- **代码：** <https://github.com/ASIG-X/JEPLO>
- **入库日期：** 2026-09-16
- **一句话说明：** PE-JEPA 学局部地形 latent，CJTS 教师—学生接到四足运动策略；强调遮挡/稀疏/噪声感知退化下的鲁棒 sim-to-real。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-16）

## 核心摘录

PE-JEPA 从原始 LiDAR + 本体状态学局部地形表征；CJTS 管线把 latent 接到 locomotion policy；评测强调退化感知条件。

**文内指标：** 多地形 sim-to-real；依赖 Unitree Go2、Mid-360 LiDAR、Isaac Lab/MuJoCo 与 Jetson 等具体条件。

## 对 wiki 的映射

- [paper-jeplo](../../wiki/entities/paper-jeplo.md)
- [12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
