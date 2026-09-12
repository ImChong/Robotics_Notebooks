# Designing physics experiments with artificial intelligence（Nature 2026 Review）

> 论文来源归档（ingest）

- **标题：** Designing physics experiments with artificial intelligence
- **类型：** paper / review / scientific-instrument-design / ai-for-science / optimization / differentiable-simulation
- **期刊：** *Nature*（2026，Review）
- **DOI：** <https://doi.org/10.1038/s41586-026-10898-6>
- **Nature 页：** <https://www.nature.com/articles/s41586-026-10898-6>
- **PDF：** <https://www.nature.com/articles/s41586-026-10898-6.pdf>
- **项目页（旗舰案例）：** <https://www.learn2design2026.com/> — NeurIPS 2026 引力波探测器设计竞赛 **Learn2Design-2026**
- **代码（竞赛 / 仿真栈）：**
  - <https://github.com/artificial-scientist-lab/Learn2Design-2026>（MIT，竞赛 starter kit）
  - <https://github.com/artificial-scientist-lab/Differometor>（MIT，JAX 可微干涉仪仿真器，PyPI `differometor`）
  - <https://github.com/artificial-scientist-lab/Differometor-Benchmark>（`dfbench` 评测框架）
  - <https://github.com/artificial-scientist-lab/GraviTune-Dataset>（~30k 探测器设计数据集）
- **机构：** 图宾根大学（University of Tübingen）/ Feyer / Zuse School ELIZA / 耶拿大学 / Nikhef / Caltech 等（竞赛团队）；Review 作者含 Tübingen、Caltech、CERN、芝加哥大学等
- **通讯作者：** Mario Krenn 等（Review）；竞赛联系见项目页
- **入库日期：** 2026-09-12
- **一句话说明：** *Nature* **Review** 将 **物理实验设计** 表述为在巨大硬件配置空间上、受实验约束的 **寻优问题**，围绕四条主线组织文献：（1）表达性搜索空间工程；（2）快速可靠仿真器；（3）科学目标→可计算目标函数；（4）能同时探索离散与连续设计选择的 AI 方法；并以 **Learn2Design-2026**（可微 **Differometor** + ~200 维连续参数 + 4h 评测预算 + 隐藏拓扑）作为引力波探测器 **de novo** 设计的当代标杆案例。

## 核心摘录（面向 wiki 编译）

### 1) 问题框架：从调参到 de novo 发现

- **要点：** 物理学进步长期依赖人类专家构思实验；AI 设计方法正从「调少数参数」走向「提出全新实验布局」。发现的配置常挑战既有设计惯例，却能匹配或超越人类方案。Review 将实验设计放在 **表达性搜索空间 × 可行约束 × 可计算目标 × 探索算法** 的交叉点上，并强调 **可计算性、实验可行性、可解释性、解可靠性** 之间的权衡。
- **对 wiki 的映射：** [`wiki/entities/paper-designing-physics-experiments-with-ai.md`](../../wiki/entities/paper-designing-physics-experiments-with-ai.md)

### 2) 四条组织性问题（Review 主干）

- **要点：**
  1. **搜索空间工程** — 如何把真实仪器参数化（连续反射率/功率/间距 vs 离散拓扑分支）且保持物理可行？
  2. **仿真器** — 需要足够快以支撑外层优化/学习，又足够可靠以预测灵敏度与噪声；**可微仿真**（JAX 等）使梯度/Hessian 方法可行。
  3. **目标函数** — 把「科学问题」（如应变灵敏度、量子噪声、功率约束）编译为可优化标量/约束。
  4. **探索方法** — 贝叶斯优化、进化、强化学习、生成模型、混合梯度-采样策略等，需处理 **离散+连续** 混合动作空间。
- **对 wiki 的映射：** 同上实体页「核心原理 / 四条主线」

### 3) Learn2Design-2026：引力波探测器设计竞赛（旗舰落地）

- **要点：** **NeurIPS 2026 Challenge**；参赛方提交 **优化算法类**（非固定设计），组织方在标准 **H100 VM** 上对 **10 个隐藏 UIFO 拓扑** 各跑 **4h wall-clock**；约 **200** 个连续自由度（激光功率、镜面反射率、网格间距等）；排名指标为 **10 次运行可行最优损失的算术平均**（越低越好）。提供 **Differometor**（纯 JAX 目标、支持梯度/Hessian）、**~30,000** 高质量设计（EuroHPC **360,000 GPU·h** 探索）、Round 1 已公开 43 队成绩。**EUR 25,000** 奖金（SPRIND 赞助）。
- **对 wiki 的映射：** 同上实体页「Learn2Design 案例」；[`sources/sites/learn2design_2026.md`](../sites/learn2design_2026.md)

### 4) Differometor 可微仿真器

- **要点：** 频域干涉仪 **JAX** 仿真器，设计对齐 **Finesse**；支持平面波传播、信号调制、量子噪声、光机效应；GPU 上自微分优化相对 Finesse+数值微分可达 **~160×** 加速（官方 README 图示）。PyPI 包 `differometor`；MIT 开源。
- **对 wiki 的映射：** [`sources/repos/artificial_scientist_lab_differometor.md`](../repos/artificial_scientist_lab_differometor.md)

### 5) 开源边界（项目页 / GitHub 核查，截至 2026-09-12）

- **已开源：** `Learn2Design-2026`（starter kit、Round 1 评测数据、提交门户链到 Codabench）；`Differometor`（MIT + PyPI）；`Differometor-Benchmark`（`dfbench`）；`GraviTune-Dataset`；竞赛站 `learn2design2026.com` 链到上述资源。
- **边界：** Nature Review 正文 **非** 单一代码仓库；复现应跟 **竞赛仓 + Differometor** 而非 PDF。隐藏评测拓扑与最终榜单由组织方持有；参赛需遵守 NeurIPS / SPRIND 资格（README 列明制裁与地域限制）。
- **利益冲突（Nature 披露）：** J.K.、S.A.、M. Krenn 正创立 Feyer GmbH（AI 工业发明），并获 SPRIND Next Frontier AI Challenge 资助；活动始于稿件提交之后，未资助本 Review 所述工作。
- **对 wiki 的映射：** [`sources/repos/artificial_scientist_lab_learn2design_2026.md`](../repos/artificial_scientist_lab_learn2design_2026.md)

## 当前提炼状态

- [x] Nature 页 / DOI / 项目页 / GitHub 开源核查
- [x] wiki 映射：`wiki/entities/paper-designing-physics-experiments-with-ai.md` 新建
