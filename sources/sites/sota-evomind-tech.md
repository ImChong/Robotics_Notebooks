# VLA SOTA Leaderboard（sota.evomind-tech.com）

> 来源归档（ingest）

- **标题：** VLA SOTA Leaderboard（兼 Dexterous Manipulation 榜）
- **类型：** site / leaderboard / benchmark-index
- **链接：** <https://sota.evomind-tech.com/>
- **代码：** <https://github.com/MINT-SJTU/Evo-SOTA.io>（MIT；GitHub Pages 静态导出）
- **维护：** EvoMind Tech & MINT-SJTU（项目负责人 Lin Tao / @EvoMind）
- **联系：** `business@evomind-tech.com`；GitHub Issues
- **入库日期：** 2026-07-23
- **一句话说明：** 社区维护的 **VLA + 灵巧手** 多基准排行榜：从已发表论文/仓库**摘录分数**（不重跑实验），按开源状态与评测协议分类展示，覆盖 LIBERO / Meta-World / CALVIN / RoboTwin 2.0 / LIBERO Plus / RoboChallenge / RoboCasa-GR1-Tabletop 与 Adroit / DexArt / Bi-DexHands。

## 覆盖基准（站点 README + Methodology）

### Vision-Language-Action

| 基准 | 主指标 | 说明 |
|------|--------|------|
| **LIBERO** | Average Success Rate (%) | 130 语言条件操作任务 |
| **LIBERO Plus** | Average Success Rate (%) | 相机/机器人/语言/光照/背景/噪声/布局等扰动维 |
| **Meta-World** | Average Success Rate (%) | MT50 多任务操作 |
| **CALVIN** | Average Length（主看 ABC→D） | 长程语言条件桌面任务 |
| **RoboTwin 2.0** | Hard Success Rate (%) | 强域随机化双臂 |
| **RoboChallenge** | Score | 真机多样家庭任务 |
| **RoboCasa-GR1-Tabletop** | Average Success Rate (%) | GR-1 桌面家庭任务 |

### Dexterous Manipulation（`/dex/`）

| 基准 | 主指标 |
|------|--------|
| **Adroit** | Mean Success Rate (%) |
| **DexArt** | Mean Success Rate (%) |
| **Bi-DexHands** | Mean Success Rate (%) |

## 排名与分类规则（Methodology 摘要）

- **数据来源：** 已发表论文与官方仓库；**不重跑**实验。
- **排序：** 各榜按主指标排序；跨基准分数**不可直接比**。
- **开源分类：** 默认展示带 Open Source 徽章的条目；无公开代码或采集截止前仍为 Coming Soon 的条目需打开 “Include All Models”。
- **方法标签：** 可按 SFT / RL 等训练范式过滤，降低协议混比。
- **免责：** 协议、种子与实现差异会导致数字漂移；引用前须回原文核实。

## 工程实现要点

- **栈：** Next.js 14（App Router）+ Tailwind + Recharts；静态导出部署 GitHub Pages。
- **数据：** `public/data/*.json`（如 `metaworld.json`）由 CSV→JSON 脚本生成；含 `standard_opensource` / `standard_closed` / `non_standard` 分区。
- **功能：** 可排序表、时间散点、模型模糊搜索（`/models/`）、中英双语、Dex 独立分区。

## 与 FabriVLA / Evo-1 的交叉（入库日核查）

截至 **2026-07-23**，Meta-World 榜 `standard_opensource` 中：

| 模型 | average | rank（该分区） | 代码 |
|------|---------|----------------|------|
| **FabriVLA** | **90.0** | 1 | <https://github.com/Youi-FabriX/FabriVLA> |
| **Evo-1** | **80.6** | 6 | <https://github.com/MINT-SJTU/Evo-1> |

## 对 wiki 的映射

- 升格 [`wiki/entities/vla-sota-leaderboard.md`](../../wiki/entities/vla-sota-leaderboard.md)
- 与 [VLA](../../wiki/methods/vla.md)、[Evo-1](../../wiki/entities/paper-evo1-lightweight-vla.md)、[FabriVLA](../../wiki/entities/paper-fabrivla.md)、[具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) 交叉

## 为何值得保留

- **选型导航：** 一次浏览多基准相对位次，再跳回论文协议细节。
- **开源可发现性：** 明确区分可复现条目与闭源/未发代码条目。
- **与本库主线对齐：** LIBERO / Meta-World / RoboTwin 正是站内 VLA 实体页高频评测语境。
