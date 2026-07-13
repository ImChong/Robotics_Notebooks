# ViDiHand — The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction（arXiv:2606.30308）

> 论文来源归档（ingest）

- **标题：** The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction
- **类型：** paper / computer-vision / hand-pose / egocentric / video-diffusion / perception / embodied-ai
- **arXiv：** <https://arxiv.org/abs/2606.30308> · PDF：<https://arxiv.org/pdf/2606.30308.pdf>
- **项目页：** <https://ACE-ViDiHand.github.io>（亦见 <https://vidihand.github.io>）
- **机构：** 南洋理工大学（NTU，Yuxi Wang, Wenqi Ouyang, Tianyi Wei, Zhiwei Zeng, Zhiqi Shen†, Xingang Pan†）；上海交通大学（SJTU，Chengkai Jin, Yufei Liu, Siyuan Huang）
- **入库日期：** 2026-07-13
- **一句话说明：** **ViDiHand** 首次将 **预训练视频 diffusion（Wan2.1-VACE）** 的内部表征用于 **第一视角双手 4D MANO 重建**：仅微调 **VACE 分支** 的 **hand-overlay rendering**（2D 骨架 → MANO mesh 两阶段课程）保留世界先验；**双分支 decoder**（hand-token + joint-heatmap + 互注意力 + 闭式 in-plane 平移求解）从单层中间特征读出 **metric-scale 双手轨迹**——**无需 detector、motion infiller 或 test-time optimization**；在 **ARCTIC / HOT3D / HOI4D** 上大幅领先 **WiLoR、OmniHands、HaMeR** 等基线（例：ARCTIC FAcc **0.997**、Jitter **3.18** mm/frame²）。

## 核心摘录（面向 wiki 编译）

### 1) 范式：从 hand-centric 到 video generative prior

- **要点：** 现有路线分两类：**图像式**（HaMeR、WiLoR 等）依赖上游 **手部检测器**，重度遮挡下检测失败即重建失败；**视频 式**（OmniHands 跨帧注意力、Dyn-HaMR / HaWoR 运动先验/infiller）仍只在 **稀缺 MANO 标注** 上学时序，难以建模 **遮挡推理、手–物交互与运动动力学**。互联网规模 **video generative models** 为合成连贯视频必须隐式掌握 **时空一致、3D 几何、遮挡补全**——与 4D hand recovery 同构。ViDiHand 是 **首个** 将此类先验用于 **egocentric 双手 4D 重建** 的工作。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-vidihand.md`](../../wiki/entities/paper-vidihand.md)
  - [`wiki/methods/wilor.md`](../../wiki/methods/wilor.md)（per-frame 强基线对照）

### 2) Hand-overlay rendering 适配 VACE

- **要点：** 骨干 **Wan2.1-VACE 1.3B**；**冻结 base DiT**，仅微调 **VACE 分支**。监督目标：在输入 clip 上 **alpha-blend 半透明手部 overlay**（含 **全遮挡帧**），标准 **flow-matching loss**，**无 MANO 参数监督**。课程：**Stage 1a** 2D joint skeleton overlay（利用 EgoDex 等大规模关节监督）；**Stage 1b** MANO mesh overlay（对齐 decoder 消费的 MANO 表面）。推理特征取 **第 15 层**、去噪步 **τ≈0.7** 的 VACE 激活（21 latent frames / 81-frame clip）。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-vidihand.md`](../../wiki/entities/paper-vidihand.md)
  - [`wiki/methods/mimic-video.md`](../../wiki/methods/mimic-video.md)（同为 video diffusion 表征下游任务）

### 3) Dual-branch decoder 与 metric 平移

- **要点：** **Hand-token 分支**：2 个 slot query 跨注意力读全手，回归 **MANO 朝向/姿态/形状 + 深度**。**Joint-heatmap 分支**：每关节 spatial softmax 得 2D anchor 与 pooled descriptor。**互注意力** 融合两路证据；**mixed-projection head** 将平移拆为 **回归深度 + 对 heatmap 的闭式 (t^x, t^y) 最小二乘**，避免 root translation/pose 歧义。损失：**L_MANO + L_cam + L_img + L_vis + L_temp**（含 on-screen BCE 抑制幻觉手）。
- **对 wiki 的映射：** 同上实体页

### 4) 评测：penalty protocol 与 SOTA

- **要点：** 基准 **ARCTIC**（双手–物重度遮挡）、**HOT3D**（鱼眼 HDR 快速运动）、**HOI4D**（训练外 cross-dataset）。采用 **penalty protocol**：漏检手以 **identity MANO @ 相机原点** 计入误差，避免「跳过硬帧」偏置。相对 **WiLoR / OmniHands / HaMeR / WildHands / Dyn-HaMR / HaWoR / InterWild / Hamba**，ViDiHand 在 **检测、3D pose、朝向/平移、时序 jitter** 四类九指标全面领先（项目页：ARCTIC MPJPE-p **21.7 mm**、PA-MPJPE-p **9.8 mm**、Jitter **3.18**）。
- **对 wiki 的映射：**
  - [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)（手部运动为模仿/策略监督信号）
  - [`wiki/overview/ego-category-01-data-collection.md`](../../wiki/overview/ego-category-01-data-collection.md)

### 5) 具身 AI 含义

- **要点：** 论文强调高质量 **egocentric hand motion** 直接影响 **从人类视频 scale dexterous manipulation / imitation** 的效果；ViDiHand 为 **in-the-wild 可扩展手部标注采集** 提供新路线——**全帧端到端、无 TTO**，利于大规模数据管线。
- **对 wiki 的映射：**
  - [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)
  - [`wiki/entities/paper-egowam-egocentric-human-wam-co-training.md`](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)

## 局限（论文自述 / 隐含）

- 依赖 **Wan2.1-VACE** 算力与 **hand-overlay** 适配数据管线；decoder 不反传 diffusion 骨干。
- 评测聚焦 **egocentric 双手 MANO**；与 **全身跟踪 / 机器人重定向** 的接口仍待系统集成。
- **HOI4D** 为 held-out；in-the-wild 案例（OakInk、Ropedia）为定性展示。

## 当前提炼状态

- [x] 摘要与主方法摘录
- [x] wiki 页面映射确认
