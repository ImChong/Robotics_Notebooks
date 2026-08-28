# Video2DoorTraversal 项目页

- **论文：** [arXiv:2608.20251](https://arxiv.org/abs/2608.20251)（v1：<https://arxiv.org/abs/2608.20251v1>）
- **项目页：** <https://video2doortraversal.github.io/>
- **机构：** 上海交通大学（SJTU）；山东大学（SDU）；纽娲机器人（NeoWa Robotics）
- **联系：** tangxincheng@sjtu.edu.cn；ryang2@sjtu.edu.cn
- **入库日期：** 2026-08-22
- **复核日期：** 2026-08-28

## 开源核查（步骤 2.5）

- 项目页顶部导航与按钮：**Paper** 指向 arXiv；**Code Coming soon**。Footer / Resources 未列 GitHub、Hugging Face、Zenodo 或权重下载。
- 页面宣称指标与论文摘要一致：五扇真门平均成功率 **96.57%**（169/175）、结构相近未见门 zero-shot **80.95%**、全程约 **13 s**。
- 资产生成对照表（DoorTwin vs PhysX-Omni / Articraft / Articulate-Anything）与论文 Table II 数字一致。
- **结论（2026-08-28）：** **待发布 / 宣称将开源**。无可运行官方仓，不建 `sources/repos/`。

## 页面要点

- 单 RGB 视频 → DoorTwin 关节门孪生 → 仿真闭环技能程序 → ArticuACT 双深度闭环 → 轮足移动操作真机。
- 真机把手类型展示：lever、doorknob、竖直圆柱、竖直方形。
- 连续 10/10 真机穿越演示；zero-shot 强调「无目标门再生成轨迹、无真机微调」。

## 交叉

- [`sources/papers/video2door_traversal_arxiv_2608_20251.md`](../papers/video2door_traversal_arxiv_2608_20251.md)
- [`wiki/entities/paper-video2door-traversal.md`](../../wiki/entities/paper-video2door-traversal.md)
