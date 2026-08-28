# WALL-SS Project Page（自变量机器人）

> 来源归档

- **标题：** WALL-SS — Scaling Long-horizon World Models via Next-Scale Autoregression
- **类型：** site / project page
- **URL：** <http://x2robot.com/pages/ss>（英文：<https://x2robot.com/en/pages/ss>）
- **论文 PDF：** <https://github.com/X-Square-Robot/wall-ss/blob/main/wall-ss-paper.pdf>
- **代码：** <https://github.com/X-Square-Robot/wall-ss>
- **机构：** 自变量机器人（X Square Robot）
- **入库日期：** 2026-08-28
- **一句话说明：** 官方项目页展示 Observation+Action 条件、next-scale 粗到细生成与闭环 rollout；与 GitHub 互指。页面为前端渲染，静态抓取几乎无正文——开源判断以 GitHub README / TODO 为准。

## 开源核查（2026-08-28）

| 入口 | 状态 |
|------|------|
| Homepage | 已挂链 — <http://x2robot.com/pages/ss> |
| Code | 已挂链 — [X-Square-Robot/wall-ss](https://github.com/X-Square-Robot/wall-ss)（MIT 许可文件已在，**训练/推理代码未发布**） |
| Paper | GitHub `wall-ss-paper.pdf`；公开 arXiv abs 入库日未见 |
| Checkpoints | 项目页与 README **未列** HF / 权重 URL |

## 页面内容要点

- **Condition** — Observation + Action；示例任务 *pour water* 与动作块 \(A_{t:t+T}\)
- **定位** — 动作可控、长程机器人仿真，而非单 clip 文生视频
- **交叉** — README 另链 InfinityStar（arXiv:2511.04675）作为 next-scale 视频骨干

## 对 wiki 的映射

- 论文摘录：[`sources/papers/wall_ss_x_square_2026.md`](../papers/wall_ss_x_square_2026.md)
- 代码归档：[`sources/repos/wall-ss.md`](../repos/wall-ss.md)
- 沉淀 **[`wiki/entities/paper-wall-ss.md`](../../wiki/entities/paper-wall-ss.md)**
