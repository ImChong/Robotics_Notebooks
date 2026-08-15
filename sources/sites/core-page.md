# CoRe-page（Humanoids 2025 项目页）

> 来源归档（ingest）

- **标题：** CoRe: A Hybrid Approach of Contact-aware Optimization and Learning for Humanoid Robot Motions
- **类型：** site / project-page
- **URL：** <https://tmjeong1103.github.io/CoRe-page/>
- **备用路径：** <https://tmjeong1103.github.io/CoRe/>（同题摘要与 BibTeX）
- **入库日期：** 2026-08-15
- **配套论文：** Humanoids 2025，pp. 293–300，DOI [10.1109/Humanoids65713.2025.11203055](https://doi.org/10.1109/Humanoids65713.2025.11203055)
- **配套代码：** <https://github.com/tmjeong1103/CoRe> — 归档见 [`sources/repos/core_retarget.md`](../repos/core_retarget.md)

## 一句话摘要

高丽大学 / KIST / UIUC 的 **Contact-aware motion Refinement（CoRe）** 官方站点：展示「文本生成人体运动 → 机型重定向 → 接触约束优化精炼 → 接触感知奖励 RL」管线，以及全身 / 轮式 / 上身三类人形上的仿真与真机视频。

## 公开信息要点（截至入库日）

- **录用：** 2025 IEEE-RAS 24th International Conference on Humanoid Robots（Humanoids），Seoul
- **作者：** Taemoon Jeong†、Yoonbyung Chai†、Sol Choi、Jaewan Bak、Chanwoo Kim、Jihwan Yoon、Yisoo Lee、Jongwon Lee、Kyungjae Lee、Joohyung Kim、Sungjoon Choi\*（†共同一作）
- **代码：** 项目页本身以论文/视频为主；**可运行实现**以 2026-08-12 的 [CoRe v0.1.0](https://github.com/tmjeong1103/CoRe/releases/tag/v0.1.0) 为准（重定向与接触精炼；RL 训练未随仓发布）
- **无 arXiv：** 截至 2026-08-15 未检索到公开预印本；可读入口为项目页摘要 + IEEE Xplore

## 方法四步（项目页 Proposed Method）

1. **Contact Segment Detection** — 由趾轨迹识别可靠足–地接触 \(\mathcal{C}_f\)
2. **Contact-Constrained Trajectory Optimization** — 消脚滑与浮空，平滑基座
3. **Feet Orientation Adjustment** — 优化支撑相足偏航
4. **Collision-handling and Smoothing** — 自碰位置修正 + 轨迹平滑

## 为何值得保留

- **非 PDF 证据：** 三类具身迁移视频与 (A) 精炼 / (B) RL 两段系统图，补 IEEE 付费墙。
- **三角溯源：** 项目页 ↔ Humanoids 论文 ↔ CoRe 软件仓。

## 对 wiki 的映射

- [`wiki/entities/paper-core.md`](../../wiki/entities/paper-core.md)
- [`wiki/entities/core-retarget.md`](../../wiki/entities/core-retarget.md)
