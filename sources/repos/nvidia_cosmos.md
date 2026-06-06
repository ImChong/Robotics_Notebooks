# NVIDIA/cosmos（Cosmos 开放平台）

> 来源归档

- **标题：** NVIDIA Cosmos
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/cosmos>
- **Stars：** ~9.4k（ingest 快照）
- **入库日期：** 2026-06-06
- **一句话说明：** NVIDIA **Physical AI 世界模型开放平台**：托管 **Cosmos 3** 全模态模型族、推理 cookbook、Diffusers / vLLM-Omni / NIM 集成与（即将发布的）微调配方。

## 仓库定位（README）

> NVIDIA Cosmos is an open platform of world models, datasets, and tools that enables developers to build Physical AI for robots, autonomous vehicles, smart infrastructure, and more.

## Cosmos 3 模型族（README 表）

| 模型 | 规模 | 主要能力 |
|------|-----:|----------|
| [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | 16B | 紧凑全模态：理解、仿真、未来预测、动作推理 |
| [Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) | 64B | 前沿规模全模态 |
| [Cosmos3-Super-Text2Image](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) | 64B | 高保真 T2I |
| [Cosmos3-Super-Image2Video](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) | 64B | 时序一致 I2V |
| [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | 16B | DROID 操纵 VLA 策略 |

## 双运行时面

| Surface | 输入 | 输出 | 典型用途 |
|---------|------|------|----------|
| **Reasoner** | 文本、视觉 | 文本 | Caption、时序定位、2D grounding、具身 CoT、物理合理性 |
| **Generator** | 文本、视觉、声音、动作 | 视觉、声音、动作 | T2I/T2V/I2V、带声视频、policy、正/逆动力学 rollout |

## 生成设定（README 摘录）

- 分辨率：256p / 480p / 720p（默认 480p）
- 画幅：16:9、4:3、1:1、3:4、9:16
- 帧率：10 / 16 / 24 / 30 FPS（默认 24）
- 帧数：5–300（默认 189）
- 精度：BF16；OS：Linux；GPU：Ampere / Hopper / Blackwell

## 动作条件 embodiment（节选）

| 类型 | 动作维度 |
|------|---------|
| 相机运动 | 9D |
| 自动驾驶 | 9D |
| 自我中心运动 | 57D |
| 单臂（DROID/UR/Fractal/Bridge/UMI） | 10D |
| 双臂（双 DROID） | 20D |
| 人形（AgiBot） | 29D |

## 集成路径

- **研究 / 训练：** HuggingFace **Diffusers** `Cosmos3OmniPipeline`
- **Generator Serving：** **vLLM-Omni**（`/v1/images/generations`、`/v1/videos/sync`、action modes）
- **Reasoner Serving：** **vLLM** + `vllm-cosmos3` 或 **Cosmos 3 Reasoner NIM** 容器
- **微调：** Cosmos Framework 训练配方（README 标注 Coming Soon）

## 许可

- 论文摘要：**Linux Foundation OpenMDW-1.1**（代码、权重、合成数据与 benchmark）
- 具体 checkpoint 以 Hugging Face 卡与仓库 LICENSE 为准

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Cosmos 3](../../wiki/entities/cosmos-3.md) | 主实体与能力矩阵 |
| [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md) | 上游改编来源之一（Diffusers/Cosmos） |
| [mimic-video](../../wiki/methods/mimic-video.md) | 使用 Cosmos-Predict2 系视频骨干 |
| [NVIDIA SO-101 Sim2Real](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md) | 课程 Strategy 3：Cosmos 演示视频增广 |

## 对 wiki 的映射

- 论文：[`sources/papers/cosmos3_arxiv_2606_02800.md`](../papers/cosmos3_arxiv_2606_02800.md)
- 项目页：[`sources/sites/cosmos3-project.md`](../sites/cosmos3-project.md)
- 实体页：**`wiki/entities/cosmos-3.md`**
