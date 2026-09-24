# zyt_world_arxiv_2609_21712

> 来源归档（ingest）

- **标题：** ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation
- **类型：** paper / generative-world-models / autonomous-driving / multi-camera / video-diffusion / closed-loop
- **arXiv：** <https://arxiv.org/abs/2609.21712>（PDF：<https://arxiv.org/pdf/2609.21712>）
- **项目页：** <https://zyt-aim.github.io/ZYT-World/>
- **代码：** 截至 2026-09-24 项目页仅链 **Technical Report（arXiv PDF）**，**未列 GitHub / Hugging Face / 权重**
- **机构：** ZYT AI Team
- **入库日期：** 2026-09-24
- **一句话说明：** 面向 **量产级 7 摄混合鱼眼–针孔 rig** 的 **实时可控驾驶世界模型**：投影专用 Plücker adapter + ego-motion AdaLN + 像素对齐 layout 条件；TF / CD / DMD / **RigCritic** 将 **40-step 双向 teacher** 蒸馏为 **每 latent 一步** 的因果流式生成；**TinyVAE（19M）**、W8A8 与推理引擎支撑 **双 GPU 7 视 ~720p @ 4 FPS**；**4DGS 跨轨迹对 + 可插拔隐式 memory** 处理同地点重访一致。

## 核心论文摘录（MVP）

### 1) 闭环四要求（Introduction）

- **链接：** <https://arxiv.org/pdf/2609.21712>
- **核心贡献：** 生成观测直接喂策略时须满足：**传感器可互换**（投影/分辨率/像素统计与量产 rig 一致）、**每步动作响应与场景可编辑**、**跨视/长时/跨轨迹一致**、**可部署硬件上实时**（生成+解码进控制周期）。
- **对 wiki 的映射：**
  - [paper-zyt-world](../../wiki/entities/paper-zyt-world.md)
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)

### 2) 原生异构 7 摄建模（Method §3 / 项目页 Architecture）

- **链接：** 项目页 + PDF §3
- **核心贡献：** **4× 柱面鱼眼（FoV>180°）+ 3× 针孔**，各视 **原生分辨率与宽高比（约 720p，5:1–5:4）** 联合生成；**双路相机控制**（Plücker rays + ego-motion AdaLN）；**layout adapter** 不经 VAE 编码 wireframe，实例级 box/heading/颜色/信号灯方向注入。
- **对 wiki 的映射：**
  - [paper-zyt-world](../../wiki/entities/paper-zyt-world.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 3) 因果一步蒸馏（Sec. 4）

- **链接：** PDF §4；项目页「One Step Is Enough」
- **核心贡献：** **Teacher Forcing → Causal Consistency Distillation → Self-rollout DMD → RigCritic**（**全 7 视 rig 联合评判**）；一步 AR 学生保留 teacher **>90% PSNR/SSIM**，FID/FVD/LPIPS 在 **11%** 内；生成侧 **107.7×** 于 40-step 双向 teacher（Figure 2 口径）。
- **对 wiki 的映射：**
  - [paper-zyt-world](../../wiki/entities/paper-zyt-world.md)

### 4) TinyVAE、memory 与长 rollout（Sec. 5–6 / 项目页）

- **链接：** 项目页 TinyVAE / Memory / Long Rollout
- **核心贡献：** **19M TinyVAE** 相对 **555M Wan** decoder **59.8×** 解码加速；**4DGS 新轨迹重渲染** 构造跨轨迹监督 + **零初始化 plug-in memory**（latent 隐式召回）；**bounded KV + attention sinks** 支撑 **分钟级** rollout 且 **每帧成本恒定**；memory 降 FVMD/FDD/LPIPS **12.5% / 6.2% / 10.3%**。
- **对 wiki 的映射：**
  - [paper-zyt-world](../../wiki/entities/paper-zyt-world.md)

## BibTeX（项目页 Citation）

```bibtex
@article{zytworld2026,
  title   = {ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation},
  author  = {{ZYT AI Team}},
  journal = {arXiv preprint arXiv:2609.21712},
  year    = {2026}
}
```

## 当前提炼状态

- [x] 摘要与项目页机制对齐
- [x] 开源核查（项目页无代码链）
- [x] wiki 页面映射确认
