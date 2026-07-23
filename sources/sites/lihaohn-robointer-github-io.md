# RoboInter Project Page

> 来源归档

- **标题：** RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation
- **类型：** site / project page
- **URL：** <https://lihaohn.github.io/RoboInter.github.io/>
- **论文（1.5）：** <https://arxiv.org/abs/2607.18709>
- **论文（站点/仓常用 1.0）：** <https://arxiv.org/abs/2602.09973>
- **代码：** <https://github.com/InternRobotics/RoboInter>
- **数据 / VQA / 权重：** HF `InternRobotics/RoboInter-{Data,VQA,VLM}`
- **入库日期：** 2026-07-23
- **一句话说明：** 官方主页展示 RoboInter **Manipulation Suite**：Data（230k+ / 571 场景 / 10+ 标注）、VQA、plan-then-execute VLA、半自动 Tool；交互 demo 与六类真机/泛化任务示意。页面文案仍偏 1.0 叙事，**World / CV（1.5）以 arXiv:2607.18709 为准**。

## 开源状态（项目页 + GitHub 交叉核查，2026-07-23）

| 项 | 状态 |
|----|------|
| Homepage | 已发布（Data / VQA / VLA / Tool 分区） |
| Code | 已挂链 → [InternRobotics/RoboInter](https://github.com/InternRobotics/RoboInter)（MIT） |
| Data / VQA / VLM | HF 已发 |
| VLA 权重 | 仓内 TODO / `RoboInterVLA` 占位 |
| RoboInter-World（1.5） | 论文已述；公开仓暂无独立目录 |

## 页面结构（策展）

- **RoboInter-Data** — 分割、夹爪、接触、轨迹、affordance、放置、抓取、稠密语言等
- **RoboInter-VQA** — 空间/时间理解·生成·规划
- **RoboInter-VLA** — Planner → 中间表示 → Executor；六类任务 ID/OOD
- **RoboInter-Tool** — 半自动标注 GUI
- **Citation** — 站点 BibTeX 可能仍为匿名占位；以 arXiv 为准

## 对 wiki 的映射

- 论文归档：[`sources/papers/robointer_1_5_arxiv_2607_18709.md`](../papers/robointer_1_5_arxiv_2607_18709.md)
- 代码归档：[`sources/repos/robointer.md`](../repos/robointer.md)
- 沉淀 **[`wiki/entities/paper-robointer-1-5.md`](../../wiki/entities/paper-robointer-1-5.md)**
