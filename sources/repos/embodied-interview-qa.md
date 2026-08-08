# embodied-interview-qa（WinstonJQ）

> 来源归档（repo）

- **标题：** 具身智能高频面试题库
- **URL：** <https://github.com/WinstonJQ/embodied-interview-qa>
- **Homepage：** <https://winstonjq.github.io/embodied-interview-qa/>
- **类型：** repo / interview-qa-bank / static-site
- **License：** MIT
- **维护者：** WinstonJQ
- **入库日期：** 2026-08-08
- **核查日规模：** 约 134★ / 7 forks（GitHub API）
- **一句话说明：** 中文具身智能秋招高频面试题库源码：`docs/interviews/*.md` 为八卷题库正文，渲染为 GitHub Pages 折叠式 HTML；MIT，欢迎 PR 补题。

## 开源核查（步骤 2.5）

| 维度 | 状态 |
|------|------|
| **开放程度** | **已开源** — 完整题库 Markdown + Pages HTML + 渲染工具 |
| **训练 / 推理代码** | 不适用（非模型仓） |
| **运行方式** | 打开 Pages；或本地打开 `docs/index.html` |
| **贡献** | Issue / PR；新题须来自公开面经，格式见 README |

## 仓库结构（master，入库日）

| 路径 | 作用 |
|------|------|
| `docs/index.html` | 主册（八卷目录 + 标签速查） |
| `docs/interviews/01_basics.md` … `08_*.md` | 八卷题库源 Markdown |
| `docs/interviews/*.html` | 渲染后的卷页 |
| `tools/templates/` | 学术页 / dashboard HTML 模板 |
| `notes/handcoding_research.md` | 手撕题调研笔记 |
| `CLAUDE.md` | 维护者 agent 工作流与质量 SLO |

## 八卷入口（与站点一致）

1. 通识基础 — `interviews/01_basics.html`
2. RL 算法 — `interviews/02_rl_algo.html`
3. VLA / 模仿学习 — `interviews/03_vla_il.html`
4. 世界模型 / Sim2Real — `interviews/04_world_sim.html`
5. 工程落地 — `interviews/05_engineering.html`
6. 腿足控制 / 遥操作 — `interviews/06_legged_control.html`
7. 3D 感知 / SLAM / VLN — `interviews/07_perception_nav.html`
8. LeetCode + 系统设计 — `interviews/08_coding_systemdesign.html`

## 对 wiki 的映射

- 主升格：[`wiki/entities/embodied-interview-qa.md`](../../wiki/entities/embodied-interview-qa.md)
- 站点：[`sources/sites/embodied-interview-qa-github-io.md`](../sites/embodied-interview-qa-github-io.md)
- 互补指南仓：[`sources/repos/embodied-ai-guide.md`](./embodied-ai-guide.md)、[`sources/repos/xbotics-embodied-guide.md`](./xbotics-embodied-guide.md)
