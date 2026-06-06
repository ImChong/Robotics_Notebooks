# Cosmos 3: Omnimodal World Models for Physical AI（arXiv:2606.02800）

> 来源归档（ingest）

- **标题：** Cosmos 3: Omnimodal World Models for Physical AI
- **类型：** paper / Physical AI / 全模态世界模型
- **arXiv：** <https://arxiv.org/abs/2606.02800>（PDF：<https://arxiv.org/pdf/2606.02800>）
- **技术报告（官网）：** <https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf>
- **项目页：** <https://research.nvidia.com/labs/cosmos-lab/cosmos3/>
- **代码与权重：** <https://github.com/NVIDIA/cosmos>
- **入库日期：** 2026-06-06
- **一句话说明：** NVIDIA Cosmos 第三代 **全模态世界模型** 家族：在统一 **Mixture-of-Transformers（MoT）** 架构下联合处理与生成 **语言、图像、视频、音频与动作序列**，以灵活 I/O 配置把 VLM、视频生成器、世界模拟器与 world-action 模型收束为同一 Physical AI 骨干。

## 摘要级要点

- **定位：** Cosmos 3 面向 **Physical AI**（机器人、自动驾驶、智慧基础设施等），主张 **一个可扩展的通用骨干** 同时承担 **理解、生成、仿真与动作**。
- **架构：** **MoT** = 自回归（AR）Transformer（Reasoner 路径，因果自注意力，下一 token 预测）+ 扩散 Transformer（Generator 路径，全注意力去噪图像/视频/音频/动作 token）；共享 Transformer 层、多模态注意力与统一 **3D mRoPE** 时空位置编码。
- **双运行时面：**
  - **Reasoner：** 文本 + 视觉 → 文本（感知、 grounding、物理推理、任务规划、动作预测、具身推理）。
  - **Generator：** 文本 + 视觉 + 声音 + 动作 → 视觉 + 声音 + 动作（世界生成、仿真、未来预测、合成数据、策略学习）。
- **能力矩阵（项目页）：** Vision-Language Reasoning、Image Generation、Audio-Visual Generation、Robot Policy、Forward Dynamics、Inverse Dynamics、Reasoning + Generation（先推理轨迹再生成交互视频）。
- **模型族（README）：** Cosmos3-Nano **16B**；Cosmos3-Super **64B**；专用 T2I / I2V 变体；Cosmos3-Nano-Policy-DROID **16B** 操纵策略。
- **开源与许可：** 代码、checkpoint、策展合成数据集与评测基准在 **Linux Foundation OpenMDW-1.1** 许可下发布（论文摘要口径）。
- **外部榜单（论文摘要 / 项目页）：** 技术报告撰写时，后训练模型在 **Artificial Analysis** 开源 **Text-to-Image** 与 **Image-to-Video** 排名第一；**RoboArena** 开源 **policy** 排名第一。

## 核心论文摘录（MVP）

### 1) 全模态统一与 Physical AI 骨干主张

- **链接：** <https://arxiv.org/abs/2606.02800> Abstract；项目页 Introduction
- **摘录要点：** 语言、图像、视频、音频、动作可在 **同一 MoT** 内以高度灵活的输入–输出组合流转；从而 **涵盖** 传统分立栈：VLM、视频生成、世界模拟器、world-action model。
- **对 wiki 的映射：**
  - [Cosmos 3](../../wiki/entities/cosmos-3.md) — 实体总览与能力矩阵。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 开源平台级生成式世界模型样本。

### 2) Reasoner vs Generator 双路径

- **链接：** GitHub README「Cosmos 3」；技术报告架构图
- **摘录要点：** Reasoner 用 **因果注意力** 做下一 token 预测（caption、时序定位、2D grounding、具身 CoT、物理合理性等）；Generator 用 **扩散去噪** 联合产出图像/视频/音频/动作；动作条件支持多 embodiment（相机 9D、AV 9D、自我中心 57D、单臂 10D、双臂 20D、人形 29D 等）。
- **对 wiki 的映射：**
  - [Cosmos 3](../../wiki/entities/cosmos-3.md) — MoT 与 mRoPE 表。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — policy / forward / inverse dynamics 作为 Joint WAM 平台能力。

### 3) 工程集成：Diffusers / vLLM-Omni / NIM

- **链接：** <https://github.com/NVIDIA/cosmos> Quickstart
- **摘录要点：** 研究路径：**Diffusers** `Cosmos3OmniPipeline`；生产 Generator：**vLLM-Omni** OpenAI 兼容 `/v1/videos`；Reasoner：**vLLM** + `vllm-cosmos3` 或 **Cosmos 3 Reasoner NIM** 容器；支持 T2I/T2V/I2V/V2V、带声视频、policy / forward_dynamics / inverse_dynamics。
- **对 wiki 的映射：**
  - [Cosmos 3](../../wiki/entities/cosmos-3.md) — 部署面与 embodiment 条件表。
  - [NVIDIA SO-101 Sim2Real 动手课](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md) — 早期课程中的「Cosmos 数据增广」可对照本代统一栈。

### 4) 与 Cosmos 生态前序工作的关系

- **摘录要点：** 仓库 README 将 Cosmos 定位为 **世界模型 + 数据集 + 工具** 开放平台；前序 **Cosmos-Predict2** 等被 [mimic-video](../../wiki/methods/mimic-video.md)、[Cosmos Policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md) 等作为视频骨干或微调基底；Cosmos 3 升级为 **全模态单栈**。
- **对 wiki 的映射：**
  - [mimic-video](../../wiki/methods/mimic-video.md) — VAM 使用 Cosmos-Predict2 系骨干的历史对照。
  - [paper-shenlan-wm-11-cosmos-policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md) — 基于 Predict2 的联合架构实例。

## 对 wiki 的映射

- 主实体页：[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)
- 项目页归档：[`sources/sites/cosmos3-project.md`](../sites/cosmos3-project.md)
- 仓库归档：[`sources/repos/nvidia_cosmos.md`](../repos/nvidia_cosmos.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
