# PanoWorld: Real-World Panoramic Generation（arXiv:2607.09661）

> 来源归档（ingest）

- **标题：** PanoWorld: Real-World Panoramic Generation
- **类型：** paper / panoramic world model / trajectory-controlled video diffusion / dataset
- **arXiv abs：** <https://arxiv.org/abs/2607.09661>
- **PDF：** <https://arxiv.org/pdf/2607.09661>
- **项目页：** <https://lihaoy-ux.github.io/panoworld-page/>
- **发表日期：** 2026-07-10（arXiv v1）
- **机构：** Insta360 Research；中国科学院自动化研究所；清华大学；武汉大学；UC Merced
- **通讯作者：** Lu Qi（Project Lead）
- **入库日期：** 2026-07-15
- **一句话说明：** **PanoWorld** 利用 ERP **旋转等变** 将相机运动简化为 **固定朝向平移**，以 **Dense Panoramic Ray-Conditioning（DPRC）** 与 **Geometry-aware Memory Augmentation（GMA）** 实现可控全景视频生成与长程记忆；配套 **World360**（12 万 clip：7 万真实 UAV + 5 万 AirSim360 仿真）与 **Wan2.2-5B** 三阶段 LoRA 训练；相对 Imagine360 / Matrix-3D / OmniRoam 在分布保真与轨迹 PSNR 上显著领先，并可经 **Causal Forcing** 蒸馏实时生成。

## 摘要级要点

- **问题定义：** 全景世界模型需在 **360° ERP** 下同时保证 **几何/辐射/时序一致** 与 **轨迹可控**；现有记忆机制（3D 点、KV cache、显式 3D 重建）多继承透视假设，在 **极区畸变与旋转视点漂移** 下检索错位，长程 **scene drift** 严重。
- **核心洞察：** ERP **rotation-equivariant**——旋转主要改变畸变模式而非场景内容 → 将 **旋转视为精确几何变换**、**显式建模仅平移**，简化 motion learning 与 memory retrieval。
- **DPRC（动作建模）：** 将 latent 像素映射到 **单位球面射线** $(\theta,\phi)$，构造 **per-ray 局部 SE(3) 变换** $\mathbf{T}_{ray}$，经 **PRoPE** 注入 DiT；把生成过程建模为 **动态 light-field 演化** 而非像素光流。
- **GMA（记忆建模）：** query 与 memory bank 在 **同一 PRoPE 几何流形** 对齐；**confidence-guided gating** $c=\mathrm{clamp}(\max_j \mathrm{Softmax}(QK^\top/\sqrt{d}),0,1)$ 融合检索特征，抑制未观测区幻觉。
- **三阶段训练：** Stage 1 LoRA 全景几何适配（**Latitude-Aware Reconstruction Loss**）；Stage 2 冻结 LoRA、训 DPRC **纯平移** parallax；Stage 3 激活 GMA **memory-anchored** 合成。
- **数据引擎：** **Rotation Decoupling** → **Uniform Spatial Resampling**（常数 $\Delta d$ 而非固定帧率）→ **Illumination Filtering**；真实 **Scene-Reality**（124 条长视频、>600 万帧、GPS-INS 6-DoF）+ 仿真 **Trajectory-Precise**（AirSim360）。
- **World360：** **120,000** 高质量序列（**70k 真实 + 50k AirSim360**），**多高度 aerial 3D 轨迹**、pose、depth（仿真侧）、text、**2K** 视频；相对 360-1M / PanoWan 等强调 **multi-altitude 户外物理变化**。
- **骨干与算力：** **Wan2.2-5B** + LoRA；480p/720p 评测；完整模型 ~**4 min 48 s**/clip；**Causal Forcing + DMD 蒸馏** 后 **161 帧 / 8 s**（单 H20），相对 Matrix-3D（~16.5 min）、OmniRoam（~31 min）数量级加速。
- **主要结果（480p）：** FID **27.64**（次优 Imagine360 81.18）；PSNR$_{75-80}$ **20.92**（Matrix-3D 18.02）；GMA 消融相对 w/o GMA **+1.07 PSNR**；ViPE 重建轨迹与 GT 对齐最紧。
- **开源承诺：** 模型、训练代码与 World360 **将公开**（论文结论口径）。
- **局限：** 仍依赖预训练透视/视频先验 + LoRA 适配；实时蒸馏版有轻微质量折中；户外真实数据以 **UAV 航拍** 为主，地面/室内机器人 egocentric 覆盖有限。

## 核心论文摘录

### 1) 旋转解耦 + DPRC 平移控制

- **链接：** §4.1.1–4.1.2；Fig. 3
- **摘录要点：** 固定 heading 下 $\mathbf{R}_t=\mathbf{I}$，相机运动由平移 $\mathbf{c}_t$ 驱动；每像素射线经球面反投影与局部正交基 $\mathbf{R}_{loc}$ 构造 $\mathbf{T}_{ray}\in SE(3)$，PRoPE 编码后注入 **Action Stream**。
- **对 wiki 的映射：**
  - [paper-panoworld-real-world-panoramic-generation.md](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md) — DPRC 与 motion decoupling 机制表。

### 2) GMA 几何一致长程记忆

- **链接：** §4.1.3；Table 4 消融
- **摘录要点：** memory keys/values 与 query 共享 ray-based PRoPE；confidence gating 自适应融合 $\mathbf{F}_{base}+(g\cdot c)\cdot\mathbf{F}_{mem}$；Random Memory 基线几何破碎，验证 **3D 射线对应** 检索必要性。
- **对 wiki 的映射：**
  - [paper-panoworld-real-world-panoramic-generation.md](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md) — GMA 与长程一致性。

### 3) World360 数据与 curation 管线

- **链接：** §3；Table 1；Fig. 2
- **摘录要点：** 三阶段对齐（旋转解耦、等空间重采样、曝光过滤）；World360 = 70k 真实 Anti-Gravity UAV + 50k AirSim360；multi-altitude 3D 路径 + pose/depth/text。
- **对 wiki 的映射：**
  - [paper-panoworld-real-world-panoramic-generation.md](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md) — World360 与 prior 全景数据集对比。

### 4) 三阶段训练与实时扩展

- **链接：** §4.2；§5.4
- **摘录要点：** 分阶段冻结/解冻 LoRA、DPRC、GMA；Causal Forcing 蒸馏为 chunk-wise AR student，Rolling Forcing 推理实现交互式实时全景生成（键盘控制 demo）。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 轨迹可控全景 WM 实例。

## 对 wiki 的映射

- 主实体页：[paper-panoworld-real-world-panoramic-generation.md](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[InfiniteDiffusion / Terrain Diffusion](../../wiki/entities/paper-infinite-diffusion-terrain-diffusion.md)（同为 360°/无限场景生成相邻方向）

## 参考来源（原始）

- arXiv:2607.09661（2026-07-10）
- 项目页：<https://lihaoy-ux.github.io/panoworld-page/>
- 相关 prior：360DVD、PanoWan、Matrix-3D、OmniRoam、AirSim360、Imagine360、CamPVG
