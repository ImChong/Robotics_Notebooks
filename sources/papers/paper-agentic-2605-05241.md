# DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation

- **作者：** Zijian Zeng, Fei Ding, Huiming Yang, Xianwei Li, Yuhao Liao
- **机构：** Tsinghua University; Alibaba Group; Bengbu University; UCSI University
- **版本 / 载体：** arXiv:2605.05241v1 [cs.RO]（2026-05-03 提交，13 页，2 图，5 表）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2605.05241) / [arXiv Abs](https://arxiv.org/abs/2605.05241)
- **项目页 / 代码：** 未确认开源代码（arXiv:2605.05241v1 正文及参考文献均未列出 GitHub 仓库、独立项目页或模型权重发布链接）
- **核对范围：** arXiv:2605.05241v1 全文（Abstract、§1–§5、Fig. 1–2 文字说明、Eq. (1)–(9)、Table 1–5 与 Limitations and Future Work）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心定位与三模块集成：** 针对多指灵巧操作策略在真机部署中的视觉、动力学与传感迁移难题，DexSim2Real 将三部分集成到零样本（zero-shot）Sim-to-Real 强化学习管线中：（1）**FM-DR**（Foundation Model-Guided Domain Randomization），用视觉语言模型（VLM）作为视觉真实感评判器配合闭环 CMA-ES 优化仿真参数分布；（2）**TVCAP**（Tactile-Visual Cross-Attention Policy），用双向交叉注意力融合 RGB、指尖触觉与本体感知；（3）**PSC**（Progressive Skill Curriculum），基于大语言模型（LLM）任务分解与 $\delta$ 难度调度器组织训练。
- **FM-DR 机制与参数（§3.1）：**
  - 使用 GPT-4V（`gpt-4-vision-preview`）对比渲染图 $I_s(\boldsymbol{\xi})$ 与真机参考图集合 $\mathcal{I}_r$，按光照、纹理、几何与物理合理性给出 1–10 分真实感评分 $s(\boldsymbol{\xi})$；3 组改写提示词变体的下游成功率波动 $<1.9\%$，评分排序 Spearman $\rho = 0.86\text{–}0.91$。
  - 将参数分布 $p(\boldsymbol{\xi}; \boldsymbol{\theta})$ 建模为高斯混合分布，以 CMA-ES（种群规模 16、$\sigma_0 = 0.3$、50 代）最大化带熵正则的目标 $\mathbb{E}[s(\boldsymbol{\xi}) + \lambda H(p)]$；网格搜索 $\lambda \in \{0.1, 0.3, 0.5, 0.7\}$ 的平均真机成功率分别为 $74.1\% / 78.2\% / 76.8\% / 72.3\%$，取 $\lambda = 0.3$。
  - 优化参数覆盖摩擦系数（$0.3\text{–}1.2$）、质量缩放（$0.8\text{–}1.5$）、3 项光照参数、纹理噪声幅值与相机位姿噪声；每项任务约需 10–15 轮（每轮 20 样本，总计约 200–300 次 VLM 查询，仅增加约 2 GPU-hours 开销）。优化后 VLM 真实感评分从均匀 DR 的 $4.2/10$ 升至 $7.8/10$，摩擦系数平均误差从 $0.35$ 降至 $0.08$，分布熵仅比均匀分布下降 $18\%$。
- **TVCAP 多模态策略（§3.2）：**
  - 分别采用预训练 DINOv2 初始化的 ViT 编码 RGB $\mathbf{o}_v$、1D-CNN 编码指尖触觉 $\mathbf{o}_t$（4 指各 15 个 taxel 的 XELA 传感器）、MLP 编码本体状态 $\mathbf{o}_p$；通过带残差连接的双向 Cross-Attention 融合视觉与触觉-本体特征，最后由 Transformer Decoder 输出未来 $T$ 步动作块 $\mathbf{a}_{1:T}$，并以 PPO 优化。
  - 触觉迁移未由 VLM 批评器直接优化，而是依赖噪声随机化（$\sigma = 0.05\text{ N}$）与注意力自适应降权。
- **PSC 渐进技能课程（§3.3）：**
  - 用 LLM 将自然语言任务分解为带成功判据的子技能序列 $\mathcal{S} = \{s_1, \dots, s_L\}$；每个子技能维护难度参数 $\delta_l \in [0, 1]$，当迭代成功率 $\text{SR}_l^{(t)} > \gamma$（$\gamma = 0.8$）时按步长 $\alpha = 0.1$ 递增，各子技能达到 $\text{SR} > 0.9$ 后按前驱终态分布串联并进行端到端复合奖励微调。
- **实验平台与主要结果（§4）：**
  - **仿真与真机：** NVIDIA Isaac Sim + PhysX 5（120 Hz 控制频率，4096 并行环境，$4\times$ A100 GPU，RL 训练 48 GPU-h + FM-DR 2 GPU-h = 50 GPU-h）；真机为 Franka Emika Panda 机械臂 + 16-DoF Allegro Hand + 指尖 XELA 触觉（15 taxels/指 × 4 指）+ 双 Intel RealSense D435（前视与腕部）。
  - **主实验（Table 1，6 项任务 × 50 次真机试验 × 3 随机种子，双盲评估员 $\kappa = 0.94$）：** DexSim2Real 平均真机成功率 **78.2%**（Pick-Place $92.3 \pm 1.8\%$、Stacking $85.7 \pm 2.1\%$、Peg Insertion $78.4 \pm 2.4\%$、In-Hand Rotation $71.2 \pm 2.7\%$、Tool Use $67.8 \pm 2.5\%$、Pouring $73.5 \pm 2.6\%$），高于零样本基线 DrEureka（65.1%）、DeXtreme（59.2%）、ADR（54.2%）、RAPS（50.6%）、Vanilla DR（45.2%），也高于使用 100 条真机示教的 Act3D（66.1%）、RVT（65.1%）、PerAct（62.3%）。
  - **Sim-to-Real 差距（Table 2）：** 平均仿真成功率 $86.4\%$、真机 $78.2\%$，绝对差距 $\Delta = \mathbf{8.3\%}$（Vanilla DR 为 $28.5\%$，ADR 为 $19.2\%$）。
  - **消融与泛化（Table 3–4）：** 去掉 FM-DR 改用 Vanilla DR 降至 $65.8\%$（$-12.4\%$），改用 ADR 降至 $69.3\%$（$-8.9\%$），随机评分 + CMA-ES 仅 $67.1\%$（$-11.1\%$）；去触觉降至 $70.1\%$（$-8.1\%$，In-Hand Rotation 下降 $14.3\%$、Peg Insertion 下降 $11.7\%$）；去交叉注意力改拼接降至 $72.4\%$（$-5.8\%$）；去 PSC 降至 $68.9\%$（$-9.3\%$）；去 Eq. (2) 熵正则降至 $71.6\%$（$-6.6\%$）。10 个未见 YCB 物体上 Pick-Place 达 $85.4\%$、In-Hand Rotation 达 $63.8\%$。
- **核查边界与局限：** 截至 2026-11-27 原文未公布代码与权重；FM-DR 当前仅优化视觉与物理参数而非触觉模型；实验仅面向刚体物体与单臂灵巧手单任务设定，未覆盖可变形体、流体或双臂协同。
- **对 wiki 的映射：** [DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation](../../wiki/entities/paper-agentic-2605-05241.md)
