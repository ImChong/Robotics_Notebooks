# UniSim-SLAM 项目页（vision3d-lab.github.io/unisim-slam）

> 来源归档（ingest 配套站点）

- **URL：** <https://vision3d-lab.github.io/unisim-slam/>
- **标题：** UniSim-SLAM: Feed-Forward SLAM with Unified Sim(3) Optimization
- **机构：** Vision3D Lab, Ulsan National Institute of Science and Technology（蔚山国立科学技术院，UNIST）
- **论文：** <https://arxiv.org/abs/2608.01706> — 归档见 [`sources/papers/unisim_slam_arxiv_2608_01706.md`](../papers/unisim_slam_arxiv_2608_01706.md)
- **配套仓库（占位）：** <https://github.com/vision3d-lab/UniSim-SLAM> — [`sources/repos/unisim_slam.md`](../repos/unisim_slam.md)
- **入库日期：** 2026-09-11
- **一句话说明：** ECCV 2026 官方落地页：两视图前端 + 多视图子图后端 + 统一 Sim(3) 因子图；展示 7-Scenes / TUM RGB-D 轨迹与重建定性对比。

## 公开信息要点（截至入库日）

| 项 | 状态 |
|----|------|
| **会议标签** | ECCV 2026 |
| **Paper / 方法图** | 已展示 Abstract、Overview、Framework、Multi-level Sim(3) Factor Graph |
| **定性结果** | 7-Scenes 轨迹对比；7-Scenes / TUM RGB-D 重建对比 |
| **代码按钮** | **未列** 可运行训练/推理入口 |
| **GitHub** | 页脚或 README 链到占位仓（`coming soon`） |
| **结论** | 项目页可用于方法理解与定性结果；**复现代码待发布** |

## 页面结构速记

1. **问题叙事** — 前馈 SLAM 的视图集合依赖、两视图 vs 多视图延迟–一致性折中。
2. **Overall Framework** — \(f_{2v}\) 跟踪 + \(f_{mv}\) 子图 → 统一 Sim(3) 因子图联合优化 \(T_i\) 与 \(S_m\)。
3. **Multi-level Sim(3) Factor Graph** — view–view / view–submap / submap–submap 三类边。
4. **结果展示** — 7-Scenes 轨迹；7-Scenes + TUM RGB-D 重建（相对基线更一致）。

## 关联资料

- 论文摘录：[`sources/papers/unisim_slam_arxiv_2608_01706.md`](../papers/unisim_slam_arxiv_2608_01706.md)
- 占位仓：[`sources/repos/unisim_slam.md`](../repos/unisim_slam.md)
- Wiki 实体：[`wiki/entities/paper-unisim-slam.md`](../../wiki/entities/paper-unisim-slam.md)
