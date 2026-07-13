# PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation（arXiv:2606.28128）

> 来源归档（ingest）

- **标题：** PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation
- **类型：** paper / embodied video generation / physics-aligned world model
- **arXiv：** <https://arxiv.org/abs/2606.28128>（PDF：<https://arxiv.org/pdf/2606.28128>）
- **项目页：** <https://dagroup-pku.github.io/PhysisForcing.github.io/>
- **入库日期：** 2026-07-13
- **一句话说明：** 北大 × NVIDIA 提出的 **训练期物理对齐框架**：在 DiT 视频骨干微调时，用 **深度感知运动掩码** 聚焦操纵/接触区域，联合 **像素级点轨迹对齐（CoTracker3）** 与 **语义级 token 关系对齐（冻结视频理解编码器）**，在 **R-Bench / PAI-Bench / EZS-Bench** 与 **WorldArena / Fast-WAM** 下游策略上稳定提升物理合理性——**推理零额外开销**。

## 摘要级要点

- **问题：** 通用与机器人微调视频生成器在接触丰富操纵中仍易出现 **轨迹不连续、穿透、反重力、抓取物漂移** 等物理违例，削弱其作为世界模拟器的可信度。
- **根因（论文观察）：** 物理不稳定主要来自 **运动物体形变** 与 **交互实体间不合理的时空关联**（尤其接触阶段）。
- **方法：** **PhysisForcing** = **区域聚焦的分层物理对齐**：
  - **物理信息区域提取：** CoTracker3 稠密轨迹 + 首帧深度前景权重 → 自适应阈值得到时空掩码 $\mathbf{M}^{\mathrm{phy}}$；
  - **像素级对齐 $\mathcal{L}^{\mathrm{phy}}_{\mathrm{pix}}$：** 中间层 DiT 特征经 MLP 后，用 query–key 相似度期望坐标预测点轨迹，与参考轨迹做 **掩码 MSE**；
  - **语义级对齐 $\mathcal{L}^{\mathrm{phy}}_{\mathrm{sem}}$：** 在掩码 token 上对齐 DiT 与冻结视频理解编码器的 **token–token 相似度矩阵**；
  - **总目标：** $\mathcal{L}=\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{pix}}\mathcal{L}^{\mathrm{phy}}_{\mathrm{pix}}+\lambda_{\mathrm{sem}}\mathcal{L}^{\mathrm{phy}}_{\mathrm{sem}}$；辅助模型 **仅训练期使用**。
- **骨干与变体：** **Wan2.2-I2V-A14B**（PF-Wan14B）、**Wan2.2-TI2V-5B**（WorldArena）、**Cosmos3-Nano LoRA**（PF-Cosmos）；训练数据为 RoVid-X 子集约 **500K** clips。
- **主要数字（论文 / 项目页）：**
  - R-Bench Avg：PF-Wan14B **62.0**（+7.1% vs vanilla ft）；PF-Cosmos **63.8**（全榜最佳）
  - PAI-Bench Domain：PF-Cosmos **93.26**；PF-Wan14B **88.20**
  - WorldArena IDM 闭环：PF-Wan5B **24.0%**（基线 Wan2.2-5B **16.0%**）
  - RoboTwin 2.0 + Fast-WAM 平均成功率：**72.8%**（+4.6% vs Fast-WAM）
- **机构：** Peking University · NVIDIA

## 核心论文摘录（MVP）

### 1) 分层 + 区域聚焦的物理监督范式

- **链接：** <https://arxiv.org/abs/2606.28128> §1–3；项目页 Method Overview
- **摘录要点：** 物理合理性在操纵视频中天然 **分层**——像素级需轨迹连续与接触相容位移，语义级需交互结果一致（推则动、抓则耦）；证据高度 **局部化** 于操纵器、物体与接触面，均匀监督会稀释信号。
- **对 wiki 的映射：**
  - [PhysisForcing](../../wiki/entities/paper-physisforcing.md) — 方法总览与流程图。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 「物理一致性缺失」挑战的对照解法。

### 2) 像素级轨迹对齐（CoTracker3 + DiT 中间层）

- **链接：** 论文 §3.2；项目页 Pixel-level Physics Alignment
- **摘录要点：** 从 DiT **中间层** hidden state 经轻量 MLP 得帧特征图；首帧 query 点与后续帧 key 做 softmax 加权坐标期望得 $\hat{\mathbf{p}}_i^t$，在 $\mathbf{M}^{\mathrm{phy}}$ 内与 CoTracker3 参考轨迹监督。
- **对 wiki 的映射：**
  - [PhysisForcing](../../wiki/entities/paper-physisforcing.md) — 像素级模块表与公式索引。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 接触丰富操纵的视频模拟可靠性。

### 3) 语义级关系对齐（冻结视频理解编码器）

- **链接：** 论文 §3.3
- **摘录要点：** 将 DiT 投影特征与冻结 encoder 特征在掩码 token 上计算 **成对余弦相似度矩阵** $\hat{\mathbf{R}}$ vs $\mathbf{R}$，用 L1 对齐，迁移 **抓取–夹爪耦合、推动–物体联动** 等全局交互结构。
- **对 wiki 的映射：**
  - [PhysisForcing](../../wiki/entities/paper-physisforcing.md) — 与像素级损失的互补关系。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 物理对齐视频表征对下游 IDM / WAM 的增益。

### 4) 跨骨干评测与下游 WAM / 策略

- **链接：** 论文 §4；项目页 Quantitative Results
- **摘录要点：** 在 **Wan2.2-A14B** 与 **Cosmos3-Nano** 上相对 base / vanilla ft 一致提升；**WorldArena action-planner（IDM）** 闭环 **16.0%→24.0%**；作 **Fast-WAM** 视频骨干时 RoboTwin 2.0 平均 **+4.6%**。
- **对 wiki 的映射：**
  - [Cosmos 3](../../wiki/entities/cosmos-3.md) — PF-Cosmos 作为 Nano 后训练物理对齐实例。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 训练期物理对齐 vs 后验 preference / 几何单点约束的对照轴。

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-physisforcing.md`](../../wiki/entities/paper-physisforcing.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[manipulation 任务页](../../wiki/tasks/manipulation.md)
