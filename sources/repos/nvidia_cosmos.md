# NVIDIA/cosmos（Cosmos 开放平台）

> 来源归档

- **标题：** NVIDIA Cosmos
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/cosmos>
- **Stars：** ~11.7k（2026-09-05 再核）
- **入库日期：** 2026-06-06
- **再核日期：** 2026-09-05
- **一句话说明：** NVIDIA **Physical AI 世界模型开放平台**：托管 **Cosmos 3** 全模态模型族、推理 cookbook、Diffusers / vLLM-Omni / SGLang / NIM 集成，以及指向 **cosmos-framework** 的微调 / 蒸馏配方。

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（平台仓 + HF 权重；具体 checkpoint 以卡上 LICENSE 为准） |
| **代码** | <https://github.com/NVIDIA/cosmos> |
| **训练框架** | <https://github.com/NVIDIA/cosmos-framework> |
| **权重** | <https://huggingface.co/collections/nvidia/cosmos3> |
| **项目页** | <https://research.nvidia.com/labs/cosmos-lab/cosmos3/> |
| **产品页** | <https://www.nvidia.com/en-us/ai/cosmos/> |
| **许可** | 论文 / 产品 FAQ：**Linux Foundation OpenMDW-1.1**；第三方依赖另计 |

## 仓库定位（README，2026-09-05）

> NVIDIA Cosmos is an open platform of world models, datasets, and tools that enables developers to build Physical AI for robots, autonomous vehicles, smart infrastructure, and more.

Cosmos 3 暴露两个运行时面：

| Surface | 输入 | 输出 | 典型用途 |
|---------|------|------|----------|
| **Reasoner** | 文本、视觉 | 文本 | Caption、时序定位、2D grounding、具身 CoT、物理合理性 |
| **Generator** | 文本、视觉、声音、动作 | 视觉、声音、动作 | T2I/T2V/I2V、带声视频、policy、正/逆动力学 rollout |

## Cosmos 3 模型族（README 表，2026-09-05）

| 模型 | 规模 | 主要能力 |
|------|-----:|----------|
| [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | 16B | 默认研究与部署入口：全模态理解+生成+动作 |
| [Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) | 64B | 前沿规模全模态；教师模型 / 数据中心 |
| [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) | 4B | 边缘 / 实时策略与视觉推理（Jetson AGX Orin / Thor / RTX Pro 6000） |
| [Cosmos3-Super-Text2Image](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image) | 64B | 高保真 T2I |
| [Cosmos3-Super-Image2Video](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) | 64B | 时序一致 I2V |
| Cosmos3-Super-Text2Image-4Step / Image2Video-4Step | 64B | DMD2 蒸馏，宣称 17–25× 加速 |
| [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | 16B | DROID 操纵 VLA 策略 |
| Cosmos3-Edge-Policy-DROID | 4B | 实时 World Action Model（README 列出） |

Edge 限制：分辨率仅 256p/480p，12–30 fps，50–150 帧；当前不支持 video-to-video transfer。

## 生成设定（README 摘录）

- 分辨率：256p / 480p / 720p（默认 480p）
- 画幅：16:9、4:3、1:1、3:4、9:16
- 帧率：10 / 16 / 24 / 30 FPS（默认 24）
- 帧数：5–300（默认 189）
- 精度：BF16；OS：Linux；GPU：Ampere / Hopper / Blackwell
- 推荐 CUDA 13；NGC 基座 `nvcr.io/nvidia/pytorch:25.09-py3`（CUDA 13）或 `25.06-py3`（CUDA 12）

## 动作条件 embodiment（节选）

| 类型 | 动作维度 |
|------|---------|
| 相机运动 | 9D |
| 自动驾驶 | 9D |
| 自我中心运动 | 57D |
| 单臂（DROID/UR/Fractal/Bridge/UMI） | 10D |
| 双臂（双 DROID） | 20D |
| 人形（AgiBot） | 29D |

## 集成路径（「Choosing an Integration」）

| 目标 | 使用 |
|------|------|
| Generator 研究 | Diffusers `Cosmos3OmniPipeline` |
| Generator 生产 API | **vLLM-Omni** 或 **SGLang** |
| Generator 开箱部署 | Generator NIM（仅 T2V/I2V） |
| Reasoner 研究 | Transformers |
| Reasoner 生产 | vLLM / **TensorRT-LLM** |
| Reasoner 开箱 | Reasoner NIM |
| 可跑训练 / 评测 | **[Cosmos Framework](https://github.com/NVIDIA/cosmos-framework)** |

微调 cookbook 已发布（不再只写 Coming Soon）：Vision SFT（Nano/Super LoRA/Edge）、Policy-DROID SFT、Reasoner LLaVA-OneVision / VideoPhy-2；蒸馏走 DMD2 四步学生。入口脚本在 `cookbooks/cosmos3/**/finetune/` 与 `distill/`。

生态仓：Cosmos Curator、Cosmos Evaluator。

## 局限（README）

长分辨率 / 复杂物理输出可出现时序不一致、相机或物体运动不稳、声画错位、动作–状态不一致、物体 morphing、3D 结构不准、物理动力学不合理。安全关键控制需额外验证与 guardrail。

## 对 wiki 的映射

- 平台总览：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- 论文：[`sources/papers/cosmos3_arxiv_2606_02800.md`](../papers/cosmos3_arxiv_2606_02800.md)
- 项目页：[`sources/sites/cosmos3-project.md`](../sites/cosmos3-project.md)
- 训练框架：[`sources/repos/nvidia_cosmos_framework.md`](./nvidia_cosmos_framework.md)
- 实体页：**`wiki/entities/cosmos-3.md`**
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 2.x 配方：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
