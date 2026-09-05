# NVIDIA Cosmos 产品页

> 来源归档

- **标题：** NVIDIA Cosmos
- **类型：** site（厂商产品页）
- **来源：** NVIDIA
- **链接：** https://www.nvidia.com/en-us/ai/cosmos/
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 产品叙事：开源 Physical AI 世界基础模型 + 数据策展 / 训练 / 评测框架；当前主推 Cosmos 3 omni-model（MoT）。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)

## 开源边界（步骤 2.5）

页内 CTA「Download Models / Customize Models / Try Now / Cookbook」指向 Hugging Face、GitHub 与 hosted catalog。FAQ 写 Cosmos WFM 使用 **OpenMDW 1.1**。配套开源框架：Cosmos Curator、Cosmos Evaluator、NVIDIA/cosmos、cosmos-framework。→ **已开源（平台 + 权重；具体卡以 LICENSE 为准）**。

交叉：

- 代码：[`sources/repos/nvidia_cosmos.md`](../repos/nvidia_cosmos.md)
- 训练框架：[`sources/repos/nvidia_cosmos_framework.md`](../repos/nvidia_cosmos_framework.md)
- Cosmos 3 项目页：[`sources/sites/cosmos3-project.md`](./cosmos3-project.md)

## 页面要点（2026-09-05）

### 定位

Develop physical AI faster with leading world foundation models and open data processing, training, and evaluation frameworks.

Cosmos 3：**The Open Physical AI Foundation Model** — first omni-model with native reasoning, world and action generation；Mixture-of-Transformers。

### 四条产品用法

| 用法 | 要点 |
|------|------|
| Power Vision AI Reasoning | 当 VLM 用：物体 / 交互 / 意图；质检、公共安全、交通、物流、自动驾驶 |
| Build Policy Models | 作 WAM 骨干；在相机与 embodiment 数据上后训练策略 |
| Simulate Worlds | 可控、物理接地的世界模拟器；闭环投放多种行为 |
| Scale Synthetic Video Data | 从文本 / 图像 / 视频 / 环境声 / 动作生成「无限」合理未来 |

### 与 Omniverse / Newton 的官方分界（FAQ）

- **Omniverse**：用生成 API、SDK 与 RTX 做真实 3D 任务仿真。
- 开发者可把 **Omniverse 仿真视频** 送进 **Cosmos Transfer**，生成可控照片级合成数据。
- 「Omniverse 提供训练前后的仿真环境，Cosmos 提供生成视频数据与训练 Physical AI 模型的基础模型。」

### Cosmos 3 vs 前代（FAQ）

Cosmos 3 是 omni-model：文本、图像、视频、声音与动作。Cosmos 2.5 / 2 把感知与生成拆成独立模型，模态主要限于文本、图像、视频。

### 后训练入口（FAQ）

GitHub 上按模态提供后训练脚本；NVIDIA TAO 7 提供 coding-agent skills。也可用 Cosmos Curator + tokenizer 自建世界模型，再用 NIM 部署。

## 对 wiki 的映射

- 平台总览：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- Cosmos 3：[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)
- Transfer：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- Cookbook：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
- 解析仿真对照：[`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)、[`wiki/entities/nvidia-omniverse.md`](../../wiki/entities/nvidia-omniverse.md)
