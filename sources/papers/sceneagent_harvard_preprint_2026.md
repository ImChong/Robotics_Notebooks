# sceneagent_harvard_preprint_2026

> 来源归档（ingest）

- **标题：** SceneAgent: 3D Capture-Derived Scenes with Predictive Physics for Policy Evaluation and Training Environments
- **类型：** paper（preprint，项目页 `@misc`）
- **来源：** [Harvard Computational Robotics 项目页](https://computationalrobotics.seas.harvard.edu/SceneAgent/)（2026）；**截至入库日未挂 arXiv**
- **代码：** <https://github.com/ComputationalRobotics/SceneAgent> — 静态站；管线 **待发布** — 见 [sources/repos/computationalrobotics-sceneagent.md](../repos/computationalrobotics-sceneagent.md)
- **入库日期：** 2026-09-27
- **一句话说明：** Harvard Computational Robotics 的 **agentic Real2Sim**：多源 3D 捕获 → 语义/预测物理/关节化/**digital sisters** → USDZ 仿真环境 + **脚本演示工厂 + VLA LoRA** sim-only 微调闭环。

## 核心论文摘录（MVP）

### 1) 问题设定与系统定位（项目页 Abstract / Intro）

- **链接：** <https://computationalrobotics.seas.harvard.edu/SceneAgent/>
- **核心贡献：** 将 **3DGS / 摄影测量 / LiDAR**（及 RGB→COLMAP→3DGS）转为 **Isaac Lab、MuJoCo、Unreal** 等可用的 **Universal Scene Description (USDZ)** 场景；强调 **per-Gaussian 预测物理**、物体分解、关节化，以及 **digital sisters** 域随机化；对比生成式视频世界模型，称 **算力更低、可对特定环境/零件精确**。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [SceneAgent 论文实体](../../wiki/entities/paper-sceneagent-real2sim-capture-physics.md)
  - [Real2Sim 纵深](../../roadmap/depth-real2sim.md)

### 2) 几何—语义—物理管线（Figure 1 / 步骤 01–07）

- **链接：** 项目页 §「From Real-world Scenes into Simulation」
- **核心贡献：**
  - **语义：** GroundingDINO + SAM → 每 Gaussian 语义特征 + codebook（LangSplat 启发）。
  - **前景/背景：** 分割前景物体、裁剪 Gaussians、**predictive infill** 背景空洞。
  - **物理：** 语义 + per-Gaussian 物理模型 + **VLM** 估计摩擦、刚性、质量、密度 → **bake physics material**。
  - **结构：** 物体分解、按需 **articulation**；导出前 **VLM 审查** 各步与位姿。
  - **Digital Sisters：** 每物体生成几何/视觉微变体；场景级 **布局与光照** 随机化。
- **对 wiki 的映射：**
  - [SimFoundry 论文实体](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 同为 splat 背景 + 可操作前景，cousins vs sisters 术语对照
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) Agentic 编排与异构输入（§ Agentic Pipeline）

- **链接：** 项目页 §「Agentic Pipeline for SceneAgent」
- **核心贡献：** **Agent swarm** 处理多样 3D 格式与易错步骤（分割尺度/旋转/放置）；可渲染 Isaac Lab 视图做 **视觉 inspect & correct**；同一流程覆盖 **COLMAP + splatfacto**、互联网 photogrammetry、既有 `.ply`、World Labs Marble 等生成 3DGS；展示软体（毛巾、软管）、铰接（相机、驾驶舱、喷气引擎）等长尾资产。
- **对 wiki 的映射：**
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — episode 级 MuJoCo 孪生 vs 本工作 **场景 + USD 资产** 级

### 4) Demonstration factory 与 VLA 训练（§ Policy training）

- **链接：** 项目页 §「SceneAgent pipeline: Policy training」
- **核心贡献：**
  - 输入：**不可直接交互** 的 Gaussian 重建 + 分割 mesh + 预测物理。
  - **Demonstration factory：** 初始状态随机（桌面区域、物体尺寸、相机/基座位姿扰动）+ **物理沉降**；**脚本专家** 用仿真内 **ground-truth 位姿** 做 pick-place；**success gate** 丢弃失败 episode。
  - **观测：** mesh 前景（机器人+物体）**深度合成** 到部署环境的 **Gaussian splat 背景**。
  - **训练：** 标准 IL 格式 + 多措辞语言指令；**LoRA 微调** 预训练 VLA（页面示例 π₀.₅、GR00T 1.6）。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 5) 评测与 sim↔real（§ Evaluation in Progress）

- **链接：** 项目页 §「SceneAgent Evaluation in Progress」
- **核心贡献：** **Sim-only 微调** 后在 Franka 真机 rollout；页面称与 SimFoundry、PolaRiS 等 **相近或更好** 的 initial success rates，**10k sim episodes** 实验进行中；**完整定量与 Pearson 类协议尚未发布**（「will update soon」）。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
