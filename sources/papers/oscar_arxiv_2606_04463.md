# OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics（arXiv:2606.04463）

> 来源归档（ingest）

- **标题：** OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics
- **类型：** paper / action-conditioned video world model / cross-embodiment / policy evaluation
- **arXiv：** <https://arxiv.org/abs/2606.04463>（PDF：<https://arxiv.org/pdf/2606.04463.pdf>）
- **项目页：** <https://wuzy2115.github.io/oscar-project-page/>
- **代码：** <https://github.com/wuzy2115/oscar-public>
- **作者：** Zhuoyuan Wu（Peking University）、Jun Gao（University of Michigan, NVIDIA）
- **入库日期：** 2026-06-20
- **一句话说明：** 以 **2D 运动学骨架渲染** 作跨具身统一动作条件、经 **四阶段数据管线** 从 216 万源视频筛得 18 万高质量片段，在 **单张 GH200** 上微调 **Cosmos-Predict2.5-2B**；开环视频质量超越 14B Kinema4D 等基线，且在 **RoboArena** 七策略虚拟评测上与真机排名 **强相关**（MMRV **0.571**、Pearson **ρ +0.750**）。

## 摘要级要点

- **问题：** 动作条件视频世界模型用于真机策略评估时面临三难——训练数据场景窄、动作跟随不精确、跨具身泛化差。
- **数据管线（四阶段）：** ① 策展（5 机器人集 + 2 人类 egocentric 集，公开约 216 万集）→ ② 质量过滤（长度 ≥70 帧、静态相机、有效动作、骨架可见）→ ③ 语义去重（SigLIP 视觉聚类 + 轨迹 RMS 验证）→ ④ Qwen3-VL-30B 字幕；最终 **180,657** 集（机器人 94,830 + 人类 85,827）。
- **动作条件：** 将 URDF/MANO 前向运动学投影为 **2D 骨架线框**（无纹理），经 WAN 2.1 VAE 编码后与目标视频 latent **逐 patch 相加** 注入 DiT；首帧 RGB 锚定外观与场景。
- **训练：** 基于 **Cosmos-Predict2.5-2B**（rectified-flow DiT）；Stage 1 **15k iter** 四具身机器人；Stage 2 机器人+人类混合 **warm-start**；**单 GH200** 完成微调。
- **开环生成（Table 2，四具身均值）：** PSNR **24.24**、SSIM **0.846**、LPIPS **0.094**、FVD **7.08**、FID **15.07**、FPS **2.214**——优于 14B Kinema4D、Genie Envisioner、EnerVerse-AC 等。
- **策略评估（RoboArena，65 session × 7 DROID 通用策略）：** 骨架条件 MMRV **0.571**、Pearson **ρ +0.750**、Spearman **r +0.852**、SISR_Δ **1.73 pp**；优于 latent-action 与 mesh 条件消融。
- **开源：** 代码、数据与 checkpoint 发布（见项目页与 GitHub）。

## 核心论文摘录（MVP）

### 1) 骨架渲染作跨具身统一条件

- **链接：** §3.2；Fig. 2–3
- **摘录要点：** 对机器人 URDF 与人类 MANO 手，用 FK + 相机投影栅格化运动学树为黑底线框；与 mesh/pointmap 相比 **不绑定外观纹理**，与 latent-action 相比 **像素级对齐、动作跟随更准**；骨架 latent 与视频 latent 经独立 patch embedder 后 **相加** 送入 DiT。
- **对 wiki 的映射：**
  - [OSCAR](../../wiki/entities/paper-oscar.md) — 条件注入与跨具身动机。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 显式几何条件支路。

### 2) 四阶段数据管线与去重

- **链接：** §4；Tab. 1
- **摘录要点：** 来源含 DROID、RH20T、InternData-A1、AgiBot-Beta、AIROA-MoMa、EgoDex、EPIC-Kitchens；过滤后机器人子集 **94,830**、人类 **85,827**；去重用 SigLIP 余弦 **>0.95** 候选对 + 64 步轨迹 RMS 自适应阈值，避免「同景不同动作」误删。
- **对 wiki 的映射：**
  - [OSCAR](../../wiki/entities/paper-oscar.md) — 数据金字塔与 Mermaid 管线图。
  - [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — 异构数据混合坐标。

### 3) 开环视频质量与条件/数据消融

- **链接：** §5.2–5.3；Tab. 2–3
- **摘录要点：** 2B OSCAR 在多数指标上优于或接近最优，且快于 14B Kinema4D；消融：mesh 与 skeleton 生成指标接近，但 mesh 依赖具身 URDF、难混入人类数据；**+Human warm-start** 优于 robot-only 与从头混合（PSNR **24.24** vs **23.48**）。
- **对 wiki 的映射：**
  - [OSCAR](../../wiki/entities/paper-oscar.md) — 基准对比表与消融节。

### 4) RoboArena 虚拟策略评估闭环

- **链接：** §5.4；Fig. 1；Tab. 4
- **摘录要点：** 对 7 个开源 DROID 通用策略（π₀-flow、π₀-FAST、PG 系列等）在 65 session 上自回归 rollout；MoGe-v2 估内参、CtRNet-X 估外参；**GPT-5** 判成功与配对偏好；报告 MMRV、Pearson/Spearman 与真机 RoboArena 排名一致性；骨架条件相关性最强。
- **对 wiki 的映射：**
  - [OSCAR](../../wiki/entities/paper-oscar.md) — 策略评估协议。
  - [RoboArena](../../wiki/methods/roboarena.md) — 分布式真机评测与 WM 代理评估衔接。
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) — 虚拟沙盒路线补例。

## BibTeX（arXiv）

```bibtex
@misc{oscar2026,
  title         = {OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics},
  author        = {Wu, Zhuoyuan and Gao, Jun},
  year          = {2026},
  eprint        = {2606.04463},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.04463}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-oscar.md`](../../wiki/entities/paper-oscar.md)
- 代码归档：[`sources/repos/oscar_public.md`](../repos/oscar_public.md)
- 项目页：[`sources/sites/oscar-project-page.md`](../sites/oscar-project-page.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[RoboArena](../../wiki/methods/roboarena.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
