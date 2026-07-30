# lsh3163.github.io/prism（PRISM 项目页）

- **标题：** PRISM — Polynomial Representations for Interaction-Structured Motor Control
- **类型：** site / project-page
- **URL：** <https://lsh3163.github.io/prism/>
- **配套论文：** [PRISM（arXiv:2607.23473）](https://arxiv.org/abs/2607.23473) — 归档见 [`sources/papers/prism_arxiv_2607_23473.md`](../papers/prism_arxiv_2607_23473.md)
- **代码：** <https://github.com/lsh3163/prism> — 归档见 [`sources/repos/prism.md`](../repos/prism.md)
- **入库日期：** 2026-07-30

## 一句话摘要

密歇根大学 **PRISM** 官方项目页：展示用**因式分解多项式模块**改写本体感觉条件通路、在 RL（Humanoid-Gym / BFM-Zero）与 IL（Diffusion Policy / SmolVLA）中提升控制，以及无线性探针解释学到的物理交互特征。

## 公开信息要点（截至入库日）

- **机构 / 作者：** University of Michigan, Ann Arbor（CSE）— Seung Hyun Lee、Stella X. Yu。
- **页首卖点：** end-to-end 学习；**不加传感器**；与现有策略 backbone 兼容；同一表征跨 RL 与模仿学习。
- **导航：** arXiv · PDF · **Code**（链到 `github.com/lsh3163/prism`）· Teaser 图。
- **结果板块：**
  - **Humanoid-Gym**：生存率 MLP 51.0 / Larger MLP 52.25 / **PRISM 92.5**；线性误差 0.2099、episode length 2233.4
  - **LIBERO（任务专属 DP）**：Diffusion 63.8 / MCC-Sensorless 47.8 / MCC-Oracle 64.5 / **PRISM 91.0**（同 RGB+state；仅 Oracle 用 force）
  - **BFM-Zero tracking EMD ↓**：Nominal / Low-fric / Payload 上均优于 baseline 与 larger control
  - **SmolVLA multi-task LIBERO ↑**：平均 66.55（vs 63.50 / 64.90）；Long 套件增益最大（53.4）
- **表征分析：** joint-power / slip / contact-impulse / contact-work 探针；emergent velocity 交互项；动力学偏移在特征空间中的分离可视化。
- **BibTeX：** `@article{lee2026prism,... arXiv:2607.23473}`。

## 开源核查（步骤 2.5，2026-07-30）

- 项目页头部与按钮均有明确 **Code → https://github.com/lsh3163/prism**。
- 仓库含独立 `prism_robot` 包、单元测试、BFM-Zero / SmolVLA 补丁与复现文档 → 判定 **已开源**（上游仿真/数据/权重不随本仓分发；顶层 LICENSE 标注仍在 finalize）。

## 为何值得保留

- **非 PDF 证据：** 低摩擦跟踪视频、接触力时序与 LIBERO 成功/失败对照，比表格更直观说明「无 force 输入的柔顺」。
- **选型坐标：** 把「加宽 MLP」与「显式多项式交互」对照写清，服务策略架构选型。
- **与 arXiv / GitHub 三角互证：** 数字与仓库 `RESULTS.md` 对齐。

## 关联资料

- 论文归档：[`sources/papers/prism_arxiv_2607_23473.md`](../papers/prism_arxiv_2607_23473.md)
- 代码仓库：[`sources/repos/prism.md`](../repos/prism.md)
- Wiki：[`wiki/entities/paper-prism.md`](../../wiki/entities/paper-prism.md)
