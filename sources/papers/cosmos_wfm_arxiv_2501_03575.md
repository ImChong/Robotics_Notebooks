# Cosmos World Foundation Model Platform for Physical AI（arXiv:2501.03575）

> 来源归档（一手论文）

- **标题：** Cosmos World Foundation Model Platform for Physical AI
- **类型：** paper / Physical AI / 世界基础模型平台
- **arXiv：** <https://arxiv.org/abs/2501.03575>
- **机构：** NVIDIA
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos **第一代** 世界基础模型平台：视频策展管线 + 扩散 / 自回归预训练 WFM + 后训练示例 + 视频 tokenizer + guardrail；开源入口当时为 NVIDIA Cosmos-Predict1。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md`](../../wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md)（该 arXiv 的唯一 canonical 节点；由 Awesome 索引升格）

## 开源边界（步骤 2.5）

论文声明 pre-trained WFM 与 tokenizer 以 **NVIDIA Open Model License** 经 NVIDIA Cosmos / Cosmos-Predict1 发布。后续官方主线迁到 Predict2.5 与 Cosmos 3（见 [`nvidia_cosmos.md`](../repos/nvidia_cosmos.md)）。→ **已开源（第一代权重/代码历史入口；现维护焦点在 Cosmos 3）**。

## 核心论文摘录

### 1) WFM 定义与 Physical AI 数据瓶颈

- **摘录：** Physical AI 需要自身的数字孪生（策略）与世界的数字孪生（世界模型）。WFM \(\mathcal{W}\) 由过去观测 \(x_{0:t}\) 与扰动 \(c_t\) 预测 \(\hat{x}_{t+1}\)；观测是 RGB 视频，扰动可以是动作、随机扰动或文本。
- **对 wiki 的映射：** [`paper-sa-2501-03575-...`](../../wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md)；[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)

### 2) 预训练–后训练两段式

- **摘录：** 大规模多样视频预训练成 generalist；再用目标 Physical AI 环境的 prompt–视频对后训练成 specialist。预训练封顶数据质量；后训练数据可以少得多。
- **对 wiki 的映射：** [`nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)

### 3) 数据与 tokenizer

- **摘录：** 从约 **2000 万小时** 视频中切出约 **1 亿** 条 2–60 秒 clip；每 256 帧用 VLM 字幕。Tokenizer 做成类视频编解码器：扩散 WFM 用连续 token，自回归 WFM 用离散 token。
- **对 wiki 的映射：** 论文实体「核心原理」

### 4) 后训练任务与平台用途

- **摘录：** 相机位姿条件（可导航虚拟世界）、机器人 video–action 未来预测、自动驾驶任务。平台用途：策略评估、策略初始化、配奖励的策略训练、规划 / MPC、带深度/语义条件的合成数据 / Sim2Real。
- **对 wiki 的映射：** [Generative World Models](../../wiki/methods/generative-world-models.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

### 5) Guardrail

- **摘录：** pre-Guard 拦有害输入，post-Guard 拦有害输出（Sec. 7）。

## 对 wiki 的映射

- Canonical 实体：[`wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md`](../../wiki/entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md)
- 平台总览：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- 策展索引源（保留）：[`sources/papers/sun_awesome_wm_2501_03575_cosmos-world-foundation-model-platform-f.md`](./sun_awesome_wm_2501_03575_cosmos-world-foundation-model-platform-f.md)
