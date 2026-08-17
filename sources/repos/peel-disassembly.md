# PEEL 双盲匿名实现（anonymous.4open.science）

- **标题：** PEEL / MAB-RRT 拆解规划（审稿匿名仓）
- **类型：** repo
- **URL：** <https://anonymous.4open.science/r/peel-disassembly-meta/>
- **许可：** 匿名页未列 SPDX；论文称 C++ 扩 OMPL
- **配套论文：** [arXiv:2608.08773](https://arxiv.org/abs/2608.08773) — [`sources/papers/peel_disassembly_arxiv_2608_08773.md`](../papers/peel_disassembly_arxiv_2608_08773.md)
- **项目页：** <https://peel-disassembly.surge.sh/#code>
- **入库日期：** 2026-08-17

## 一句话说明

项目页列出三个匿名仓：meta（Docker + 子模块）、MAB-RRT planner（OMPL 插件）、robot-pipeline（五阶段 Fetch 执行）。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| Meta | <https://anonymous.4open.science/r/peel-disassembly-meta/> |
| Planner | <https://anonymous.4open.science/r/peel-mab-rrt-planner/> |
| Pipeline | <https://anonymous.4open.science/r/peel-robot-pipeline/> |
| 依赖 | Robowflex、OMPL、DARTSim、MoveIt |

最短复现：从 meta-repo 按项目页「Docker build + ready-to-run demos」起步。匿名镜像可能随审稿结束迁移到实名 GitHub。

## 与 wiki 的关系

- 实体页：[paper-peel-disassembly](../../wiki/entities/paper-peel-disassembly.md) — 含按匿名仓入口绘制的时序图。
