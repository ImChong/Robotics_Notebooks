# HumanoidArena 项目页（humanoidarena.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://humanoidarena.github.io/>
- **标题：** HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning
- **类型：** project site / benchmark
- **机构：** 香港科技大学（广州）；北京工业大学；哈尔滨工业大学（深圳）；深圳北理莫斯科大学；京东探索研究院
- **论文：** <https://arxiv.org/abs/2606.17833>
- **代码：** <https://github.com/William-wAng618/HumanoidArena>（默认分支 `release/open-source-prep`）
- **数据集：** ModelScope / HF — `Twang2026/HumanoidArenaV3.1`、`WilliamWang16/HumanoidArena_dataset_v3_1`
- **模型：** ModelScope / HF — `Twang2026/HumanoidArena_models`、`WilliamWang16/HumanoidArena_models`
- **仿真资产：** Google Drive（项目页 Assets）
- **原始数据：** ModelScope `Twang2026/HumanoidArena_raw`
- **入库日期：** 2026-07-22
- **一句话说明：** 仿真优先 egocentric 分层全身基准落地页：7 项腿关键 HOI/HSI、TWIST2/SONIC 双 GMT、四扰动轴 + Cross-GMT；已发布代码、LeRobot 数据、checkpoint、资产与 raw 数据（multicam 仍 planned）。

## 开源状态（2026-07-22 项目页 + README 核查）

| 产物 | 状态 |
|------|------|
| 代码 / 评测脚本 | **已开源** · MIT · 分支 `release/open-source-prep` |
| LeRobot 数据集 | **已发布** |
| 策略 checkpoint | **已发布** |
| 仿真资产 | **已发布**（Drive） |
| Raw 示范 | **已发布** |
| Multicam 数据 | **待发布**（Release Plan 未勾选） |

## 页面要点

- **任务：** Football / DoubleDesk / P&PBox / OpenDoor / SitSofa / Boxing / VisNavi（各任务均有 TWIST2 与 SONIC 入口）
- **管线：** PICO egocentric + GMR → 35D 共享参考 → Isaac Lab 双 GMT → NPZ → 64D state / 40D 中间动作 → LeRobot → 评测
- **评测：** Base / Semantic / Vision / Execution + Cross-GMT
- **可选：** SONIC latent64 输出模式（64D latent action，非默认 40D semantic）

## 对 wiki 的映射

- [HumanoidArena 实体](../../wiki/entities/paper-humanoidarena.md)
- [代码归档](../repos/humanoidarena.md)
- [论文摘录](../papers/humanoidarena_arxiv_2606_17833.md)
