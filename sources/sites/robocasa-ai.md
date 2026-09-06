# RoboCasa / RoboCasa365 项目站

- **标题：** RoboCasa — Large-Scale Simulation of Everyday Tasks
- **类型：** site / 官方项目页
- **链接：** https://robocasa.ai/
- **文档：** https://robocasa.ai/docs/introduction/overview.html
- **Leaderboard：** https://robocasa.ai/leaderboard.html
- **代码：** https://github.com/robocasa/robocasa（已开源）
- **入库日期：** 2026-09-06
- **一句话说明：** UT Austin 主导的厨房日常任务大规模仿真平台；RoboCasa365 扩至 365 任务、2500 场景与 2200+ 小时演示，并提供通才策略公开排行榜。
- **沉淀到 wiki：** 是 → [`wiki/entities/robocasa.md`](../../wiki/entities/robocasa.md)

---

## RoboCasa365 四支柱（站点 + 文档）

1. **Diverse tasks** — LLM 辅助定义 **365** 日常任务
2. **Diverse assets** — **2500+** 厨房场景、**3200+** 3D 物体
3. **High-quality demonstrations** — **600+ h** 人类演示 + **1600+ h** 合成机器人轨迹
4. **Benchmarking support** — Diffusion Policy、π₀、GR00T 等 + [Leaderboard](https://robocasa.ai/leaderboard.html)

---

## 场景与资产

- 原版 **120** 场景 → RoboCasa365 **2500** 独特厨房（50 新布局 × 50 新风格）
- 可交互家具：柜门、炉灶旋钮、微波炉、水槽等（关节与状态变化）
- **跨具身：** 单臂移动平台、人形、带臂四足
- **生成式扩增：** MidJourney 纹理（墙/地/台面/柜门各 100）；Luma AI 等 text-to-3D 物体

---

## 十种基础技能 → 65 原子任务

Pick-place、开关门/抽屉、拧旋钮、扳杠杆、按按钮、插入、导航、滑轨、开盖等；复合任务由 **GPT-4** 按场景与技能组合（煮咖啡、补货、蒸蔬菜等）。

---

## 引用

- RoboCasa365：ICLR 2026（arXiv 见项目 PDF）
- RoboCasa 原版：RSS 2024

---

## 对 wiki 的映射

- [RoboCasa](../../wiki/entities/robocasa.md)
- [RoboCasa365 论文实体](../../wiki/entities/paper-notebook-robocasa365-a-large-scale-simulation-framework-f.md)
