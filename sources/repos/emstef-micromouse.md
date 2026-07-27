# emstef/Micromouse（Micromouse in Webots）

> 来源归档

- **标题：** Micromouse in Webots
- **类型：** repo + 项目页（课程作业：Webots 16×16 迷宫自主代理）
- **作者：** [emstef](https://github.com/emstef)
- **链接：** https://github.com/emstef/Micromouse
- **项目页：** https://emstef.github.io/Micromouse/（[站点归档](../sites/micromouse-in-webots-github-io.md)）
- **课程：** COMP513 Autonomous Agents @ ECE / TUC
- **入库日期：** 2026-07-27
- **一句话说明：** 在 Webots 中把 Micromouse 竞赛做成可学代理：定位、建图、路径规划（Flood Fill）与运动控制；基于 Rat’s Life / e-puck 改造 16×16 迷宫。
- **开源状态：** **已开源** — GitHub 含 `worlds/`、`controllers/`、`protos/`；项目页提供源码 / 演示文稿 / 报告下载。
- **沉淀到 wiki：** [Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **零硬件仿真入口**：用 Webots 学 Micromouse 四件套（localization / mapping / planning / motion），再迁真机。
- **算法叙事清晰**：搜索跑建图 → 竞速跑；强调「最快路径 ≠ 最短路径」（少转弯更快）。
- **外链经典资料**：指向 Micromouse Online、算法综述等，适合作为概念页「仿真」小节锚点。

## 技术要点（项目页）

| 项 | 说明 |
|----|------|
| 仿真器 | Webots（ODE 刚体动力学） |
| 机器人 | e-puck（改自 Rat’s Life demo） |
| 迷宫 | 16×16；可加载历史竞赛迷宫档案 |
| 规划 | **Flood Fill**（目标格权值 0，沿梯度走最短格路径） |
| 传感 | 轮式里程计 + IR 墙检 / 校正 |
| 语言 | C + Java（继承原 demo） |

> 说明：用户材料写「支持 Flood Fill、A\* 等」；本仓库项目页正文以 **Flood Fill** 为主实现。A\* / 其他搜索常见于 Micromouse 社区与 [lime7git 仿真器](lime7git-micromouse.md)，本页不夸大仓内已实现集合。

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 代码 | **已开源**（GitHub 公开） |
| 项目页 | **已发布**，与仓互指 |
| LICENSE | 入库日未见醒目 SPDX — 使用前确认 |

## 对 wiki 的映射

- [Micromouse](../../wiki/concepts/micromouse.md)
- [A*](../../wiki/methods/a-star.md)
- [足球场仿真](../../wiki/concepts/soccer-field-simulation.md)（同属 Webots 教学仿真对照）
