# ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2503.21268>
- **入库日期：** 2026-07-10
- **一句话说明：** 人体动作恢复（HMR）研究多聚焦地面动作（如跑步），对离地（off-ground）的攀岩动作研究稀少，部分因攀岩动作数据集（尤其大规模、有挑战性的 3D 标注）匮乏。作者采集 AscendMotion ——一个大规模、标注良好、有挑战性的攀岩动作数据集：41.2 万帧 RGB、LiDAR 帧与 IMU 测量，含 22 位熟练攀岩教练在 12 面不同岩壁上的攀岩动作。攀岩动作捕捉难在需精确恢复复杂姿态 + 全局位置；现有全局 HMR 方法难以胜任。为此提出 ClimbingCap，在全局坐标系下连续重建 3D 攀岩动作：关键是用 RGB 与 LiDAR 分别在相机坐标与全局坐标重建动作，并联合优化。CVPR 2025 收录。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-climbingcap-multi-modal-dataset-and-method-for-r](../../wiki/entities/paper-notebook-climbingcap-multi-modal-dataset-and-method-for-r.md).

## 对 wiki 的映射

- [paper-notebook-climbingcap-multi-modal-dataset-and-method-for-r](../../wiki/entities/paper-notebook-climbingcap-multi-modal-dataset-and-method-for-r.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate.html>
- 论文：<https://arxiv.org/abs/2503.21268>
