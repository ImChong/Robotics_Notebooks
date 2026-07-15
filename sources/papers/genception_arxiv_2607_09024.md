# Video Generation Models are General-Purpose Vision Learners

> 来源归档（ingest）

- **标题：** Video Generation Models are General-Purpose Vision Learners
- **类型：** paper / computer-vision / foundation-model / video-perception
- **来源：** arXiv abs / HTML；ECCV 2026 项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.09024>
  - <https://arxiv.org/pdf/2607.09024>
  - <https://arxiv.org/html/2607.09024v1>
  - <https://genception.github.io/>
- **作者：** Letian Wang*, Chuhan Zhang, Rishabh Kabra, Jasper Uijlings, Steven Waslander, Andrew Zisserman, João Carreira, Kaiming He, Misha Andriluka, Eduard Gabriel Bazavan, Andrei Zanfir, Cristian Sminchisescu（*Work done while at Google DeepMind）
- **机构：** Google DeepMind；University of Toronto；University College London；University of Oxford；MIT；Lund University
- **录用会议：** ECCV 2026
- **代码：** TBA（项目页标注即将开源）
- **入库日期：** 2026-07-15
- **一句话说明：** **GenCeption** 将预训练 **text-to-video 扩散骨干（WAN 2.1）** 改造成 **单步 feed-forward** 统一视频感知模型，以 **文本指令** 在 **共享权重** 下覆盖深度/法线/相机位姿/前景与指代分割/3D 人体关键点等任务，在多项 benchmark **匹配或超越任务专家**（Depth Anything 3、SAM3、D4RT、VGGT-Ω 等），并展示 **合成数据训练 → 真机/多实例/OOD 类别** 的涌现泛化。

## 核心论文摘录（MVP）

### 1) 核心主张：text-to-video 生成预训练 ≈ CV 的 next-token prediction

- **链接：** <https://arxiv.org/abs/2607.09024>（Abstract / §1）；<https://genception.github.io/>
- **摘录要点：**
  - NLP 已从任务专用模型演进到 **统一基础模型**；CV 仍停留在 **SAM / Depth Anything** 等 **任务专用** 架构阶段。
  - 通用视觉预训练需满足三要素：**时空演化先验**、**视觉–语言对齐**、**可规模化**——大规模 **text-to-video 生成** 同时满足。
  - **GenCeption** 将预训练 **视频扩散骨干** 定义为 **feed-forward 感知模型**，由 **文本指令** 切换任务；在深度、法线、相机位姿、指代分割、3D 关键点等 **多样任务** 达 SOTA 或可比专家。
  - **视频生成骨干** 在同等微调设定下优于 **V-JEPA、VideoMAE V2**；用 **7×–500× 更少训练数据** 可达 **D4RT、VGGT-Ω** 量级性能。
  - **涌现行为：** 仅在 **合成人视频** 上训练，可 **零样本** 迁移到真实视频、多实例场景与 **动物/机器人** 等 OOD 类别。
- **对 wiki 的映射：**
  - [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) — 从图像（Vision Banana）扩展到 **原生视频域** 的范式证据。
  - [GenCeption](../../wiki/entities/genception.md) — 方法、架构与 benchmark 表。

### 2) 方法：扩散 → 单步 feed-forward + RGB 统一表征 + 稀疏 token

- **链接：** <https://arxiv.org/abs/2607.09024>（§3）；<https://genception.github.io/>（Architecture）
- **摘录要点：**
  - **基座：** 开源 **WAN 2.1** text-to-video 扩散模型（VAE + 文本编码器 + DiT）；训练分辨率 **480×832**、**81 帧 @ 24 FPS**。
  - **Feed-forward 改造：** 将 **干净输入视频 latent**（非噪声 latent）送入 DiT，**固定 timestep t=0**（Rectified Flow 终止态），**单次前向**；对 DiT 输出的 velocity **取负** 以对齐目标 latent——把 DiT 当作 **强特征提取器**，取 **最后一层** 特征直连解码器。
  - **稠密任务统一：** 深度、分割、法线、DensePose、**Rothko raymap**（相机位姿 6 通道压缩进 RGB）等均在 **[0,1] RGB 空间** 监督，可在 **latent 空间** 高效算 **统一 L2 loss**（类比 LLM 单目标）。
  - **稀疏任务：** 每帧附加 **可学习 token**（3D RoPE + 时间位置插值），MLP 解码 **2D/3D 关键点**；论文指出该设计与生成预训练 **注意力机制** 冲突较大，联合训练时 **3D 关键点** 可能受损。
  - **数据：** 主训练 **7,500** 条 Blender 合成人体视频（RenderPeople × CMU mocap × HDRI/全场景）；深度/法线/分割/DensePose/关键点/相机轨迹多 pass 渲染；辅以 TartanAir、Virtual KITTI、MVS Synth；**指代分割** 混入 MeViS、Ref-COCO、YouTube-VOS 等真实数据。
  - **深度归一化：** 场景中值归一化 + $\alpha\log(d+1)$ 映射到 [0,1]，避免任务专用 scale-invariant loss。
- **对 wiki 的映射：**
  - [GenCeption](../../wiki/entities/genception.md) — Mermaid 管线、稠密/稀疏双路径。

### 3) 与 Vision Banana 及并发工作的定位

- **链接：** <https://arxiv.org/abs/2607.09024>（§2.3）
- **摘录要点：**
  - 与 [Vision Banana](../../wiki/entities/vision-banana.md)（*Image Generators are Generalist Vision Learners*）共享 **「生成预训练 + 文本指令 + RGB 输出空间」** 范式，但 GenCeption 在 **原生视频域** 捕获 **时序一致性**，并采用 **feed-forward** 而非多步采样。
  - 相对 **training-free prompting**（如 Wiedemer et al.）提供 **专用 post-training** 与 **标准化定量评测**。
  - 相对 DepthCrafter、NormalCrafter、ReferEverything 等 **单任务视频扩散改造**，本文是 **稠密+稀疏多任务统一架构**。
  - 由前人工作 **THFM**（human-centric）扩展为 **通用视觉模型**。
- **对 wiki 的映射：**
  - [Vision Banana](../../wiki/entities/vision-banana.md) — 图像域姊妹工作交叉引用。
  - [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) — 谱系表补充 **视频 feed-forward 统一感知** 分支。

### 4) 主要定量结果（Generalist-L，WAN 2.1 14B，Table 1 节选）

- **链接：** <https://arxiv.org/abs/2607.09024>（§4.4, Table 1）；<https://genception.github.io/>（Results）
- **摘录要点（Ours - Generalist - L vs 专家）：**
  - **法线 Hi4D mAE ↓：** **11.47**（优于 Sapiens 12.14、Lotus-2 30.3 等）
  - **深度 Sintel AbsRel ↓：** **0.156**（优于 Depth Anything 3 的 0.201；Generalist 略逊于 Specialist-L 0.130）
  - **深度 KITTI AbsRel ↓：** **0.048**（与 D4RT 0.051、VGGT-Ω 0.041 同级）
  - **相机位姿 Sintel ATE ↓：** **0.062**（优于 VGGT 0.168）
  - **前景分割 VideoMatte MSE ↓：** **0.0010**（优于 RVM 0.0010 持平、MODNet 0.0054）
  - **指代分割 Ref-DAVIS J&F ↑：** **75.8**（优于 ReferEverything 75.0、SAM3 41.3）
  - **指代分割 MeViS J&F ↑：** **69.0**（优于 SAM3+Gemini 57.5）
  - **数据效率（深度，Table 2）：** 14B + 7.5K 合成视频（0.9M 帧）平均 AbsRel **0.093**；加 4 数据集至 1.23M 帧后 **0.071**，用 **7×–500× 更少数据** 逼近 D4RT / VGGT-Ω。
  - **预训练消融：** 随机初始化 DiT **几乎不收敛**；逐步解冻预训练层性能单调提升；WAN 2.1 显著优于同数据上的 V-JEPA / VideoMAE V2。
- **对 wiki 的映射：**
  - [GenCeption](../../wiki/entities/genception.md) — 结果表与推理成本（1.3B：5.92s/81帧；14B：10.03s）。

### 5) 涌现泛化与联合训练教训

- **链接：** <https://arxiv.org/abs/2607.09024>（§4.5–4.6）；<https://genception.github.io/>（Emergent Behaviors）
- **摘录要点：**
  - **Sim-to-real：** 纯合成训练，真机视频上细节（猫须、发丝）可 **超越训练渲染质量**。
  - **多实例 / OOD：** 单物体合成训练 → 真实多实例；仅人类训练 → 动物、机器人、火箭等 **语言指代分割**。
  - **联合训练：** Generalist 在 **前景/指代分割** 常受益，但 **3D 人体关键点** 联合训练 **严重退化**——归因于 **token 回归** 与 DiT 原生注意力机制冲突；启示：**post-training 应最小化架构改动**，或重新设计预训练以原生支持多样输出。
  - **Grounded 4D：** 单视频预测 per-pixel 几何 + 相机位姿，可 lift 为 **4D 点云** 并做语言接地 fly-through（项目页演示）。
- **对 wiki 的映射：**
  - [视觉表征作为策略输入](../../wiki/concepts/visual-representation-for-policy.md) — 统一视频感知作为机器人/AR 上游。
  - [GenCeption](../../wiki/entities/genception.md) — 涌现行为与局限节。

## 当前提炼状态

- [x] arXiv 摘要、§1–§4 方法与 Table 1–2 主结果已摘录
- [x] genception.github.io 能力演示、架构图与涌现行为区块已交叉核对
- [x] wiki 映射：`wiki/entities/genception.md`、`wiki/concepts/generative-vision-pretraining.md`
- [ ] 代码开源后补 `sources/repos/` 与权重/推理脚本细节
