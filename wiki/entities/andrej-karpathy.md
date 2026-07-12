---

type: entity
tags: [deep-learning, education, tesla, openai, llm, methodology, computer-vision, reinforcement-learning, stanford]
status: complete
updated: 2026-07-12
related:
  - ../references/llm-wiki-karpathy.md
  - ./karpathy-autoresearch.md
  - ./llms-from-scratch-raschka.md
  - ../methods/reinforcement-learning.md
  - ../concepts/deep-learning-foundations.md
  - ../concepts/backpropagation.md
  - ../concepts/transformer.md
  - ../concepts/ai-auto-research.md
  - ../overview/robot-learning-overview.md
  - ../methods/imitation-learning.md
  - ../../roadmap/depth-vla.md
sources:
  - ../../sources/sites/karpathy-ai.md
  - ../../sources/blogs/karpathy_llm_wiki_gist.md
  - ../../sources/repos/karpathy-autoresearch.md
  - ../../sources/courses/karpathy_zero_to_hero_youtube.md
  - ../../sources/repos/nn-zero-to-hero.md
summary: "Andrej Karpathy：OpenAI 创始成员、前 Tesla AI 总监（Autopilot 视觉全栈）、Stanford CS231n 创始讲师；现以 Zero to Hero 等教育内容为主，其 LLM Wiki Gist 是本知识库维护范式的思想来源。"
---

# Andrej Karpathy

## 一句话定义

**Andrej Karpathy** 是连接 **深度学习教育、大规模视觉系统与 LLM 时代知识工程** 的关键人物：从 Stanford CS231n 与 micrograd 把神经网络讲清楚，到 Tesla 时期把 **数据标注–训练–定制芯片部署** 收成 Autopilot 视觉闭环，再到提出 **LLM 维护持久 wiki** 模式——本仓库的 ingest / query / lint 即其后者在机器人研究域的实例化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LLM | Large Language Model | 大语言模型；Karpathy LLM Wiki 模式中的维护代理 |
| CNN | Convolutional Neural Network | 卷积神经网络；CS231n 与 Autopilot 视觉栈核心 |
| RL | Reinforcement Learning | 强化学习；OpenAI 早期研究与 Zero to Hero 教程覆盖 |
| FSD | Full Self-Driving | Tesla 全自动驾驶目标；其任内 Autopilot 团队长期主线 |

## 为什么重要

- **机器人栈上的「产品级视觉 ML」样本**：Tesla 任内公开描述 **车内标注 + NN 训练 + 定制推理芯片** 一体化，是理解 **Sim2Real 以外的大规模部署工程** 与 **Software 2.0** 叙事的参照点。
- **研究–教育–工具三位一体**：CS231n、Zero to Hero（micrograd / makemore / nanoGPT）、Software 2.0 等构成社区 **深度学习入门与工程习惯** 的公共基础设施；与本站 [Deep Learning Foundations](../concepts/deep-learning-foundations.md)、[Reinforcement Learning](../methods/reinforcement-learning.md) 直接相交。
- **本知识库的方法论作者**： [LLM Wiki（Karpathy）](../references/llm-wiki-karpathy.md) 提出 **compilation beats retrieval**；Robotics_Notebooks 的 `sources/` → `wiki/` → `schema/` 分层即该思想在 **机器人跨主题知识** 上的落地。

## 核心脉络（与机器人学习相关的子集）

### 1. 早期：仿真与运动技能

- UBC MSc：物理仿真 **figures** 的学习控制器（「machine-learning for agile robotics but in simulation」）。
- 代表论文：**Locomotion Skills for Simulated Quadrupeds**（SIGGRAPH 2011）、**Curriculum Learning for Motor Skills**（AI 2012）——与 [Locomotion](../tasks/locomotion.md) 主题同源，早于近年人形 RL 热潮。

### 2. 视觉–语言与规模化训练

- Stanford PhD（Fei-Fei Li）：CNN/RNN、image captioning、Video CNN 等；**CS231n** 成为视觉深度学习课程事实标准之一。
- OpenAI（2015–2017）：创始成员；深度 RL 与生成模型早期探索。

### 3. Tesla：Autopilot 视觉全栈（2017–2022）

- **Director of AI**：统领计算机视觉——车内数据标注、神经网络训练、**定制推理芯片** 部署。
- 公开材料：Tesla AI Day 2021、Autonomy Day 2019、CVPR / ScaledML「AI for Full Self-Driving」等；与 [Pieter Abbeel](https://www.youtube.com/watch?v=GxYFaZZA_ms) *Robot Brains* 对谈连接 **学术机器人** 与 **量产视觉栈** 话语。
- 短暂参与 **Optimus** 人形项目（主页自述 *very briefly*）——与本站 [Humanoid Robot](./humanoid-robot.md) 叙事相邻但非其主贡献轴。

### 4. OpenAI 回归与合成数据（2023–2024）

- 组建 **midtraining** 与 **synthetic data generation** 团队——与当前 LLM/VLA **数据飞轮** 讨论同频。

### 5. 2024–：教育与 LLM 知识工程

- YouTube **技术轨 — [Neural Networks: Zero to Hero](../../sources/courses/karpathy_zero_to_hero_youtube.md)**（10 集 ~19 h）：micrograd → makemore → GPT → Tokenizer → GPT-2 124M；配套 [`nn-zero-to-hero`](../../sources/repos/nn-zero-to-hero.md) notebook 与 micrograd / makemore / nanoGPT 仓库。走 [VLA 纵深](../../roadmap/depth-vla.md) 者可作 **动手前置**，与 [LLMs-from-scratch（Raschka）](./llms-from-scratch-raschka.md) 二选一或交叉。
- **大众轨 — LLM 科普**：*Intro to LLMs*、*Deep Dive into LLMs* 等（见推荐继续阅读）。
- **LLM Wiki Gist**：持久 markdown wiki + ingest/query/lint —— 见 [LLM Wiki 方法论页](../references/llm-wiki-karpathy.md)。
- **autoresearch（2026）**：[karpathy/autoresearch](./karpathy-autoresearch.md) — 在 nanochat 单 GPU 栈上让编码代理通宵改 `train.py`、以 **val_bpb** 与固定 5 分钟预算循环实验；人类迭代 `program.md` 作为「研究组织技能」。与 [AI Auto-Research](../concepts/ai-auto-research.md) S3 阶段直接同构。

## 流程总览（Tesla 期公开描述的视觉 ML 闭环）

下列为 Karpathy 多次演讲归纳的 **逻辑骨架**（细节以 Tesla 官方材料为准），与机器人 **数据–训练–部署** 飞轮同构：

```mermaid
flowchart LR
  Fleet["车队 / 传感器数据"] --> Label["车内标注\n与数据引擎"]
  Label --> Train["NN 训练\n大规模 GPU"]
  Train --> Chip["定制推理芯片\n车载部署"]
  Chip --> Fleet
```

## 常见误区或局限

- **个人页 ≠ 实时新闻源**：任职、项目状态以机构公告与一手演讲为准。
- **LLM Wiki 是模式文件，非本仓库配置**：具体目录、lint 规则、CI 门禁见 [Ingest Workflow](../../schema/ingest-workflow.md) 与 [AGENTS.md](../../AGENTS.md)。
- **Tesla / Optimus 贡献边界**：主页对 Optimus 仅一笔带过；人形控制主线应优先引用 LECAR / GEAR 等当代论文簇，而非将其归因于 Karpathy 个人。

## 关联页面

- [LLM Wiki 方法论（Karpathy）](../references/llm-wiki-karpathy.md)
- [autoresearch（karpathy/autoresearch）](./karpathy-autoresearch.md)
- [AI Auto-Research](../concepts/ai-auto-research.md)
- [Robot Learning Overview](../overview/robot-learning-overview.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Deep Learning Foundations](../concepts/deep-learning-foundations.md)
- [Backpropagation](../concepts/backpropagation.md)
- [Transformer](../concepts/transformer.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
- [Humanoid Robot](./humanoid-robot.md)
- [Ingest Workflow](../../schema/ingest-workflow.md)

## 参考来源

- [Karpathy 个人站点原始资料](../../sources/sites/karpathy-ai.md)
- [Zero to Hero 播放列表（YouTube）](../../sources/courses/karpathy_zero_to_hero_youtube.md)
- [nn-zero-to-hero 配套仓库](../../sources/repos/nn-zero-to-hero.md)
- [LLM Wiki Gist 原始资料](../../sources/blogs/karpathy_llm_wiki_gist.md)
- [autoresearch 仓库原始资料](../../sources/repos/karpathy-autoresearch.md)

## 推荐继续阅读

- [karpathy.ai](https://karpathy.ai/) — 职业时间线、演讲与项目总索引
- [LLM Wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — 方法论原文
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — 自主 LLM 训练实验环
- [Neural Networks: Zero to Hero（YouTube，10 集）](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — micrograd → makemore → GPT → GPT-2 124M
- [nn-zero-to-hero（GitHub）](https://github.com/karpathy/nn-zero-to-hero) — 各讲 notebook 归档
- [LLMs-from-scratch（Raschka）](./llms-from-scratch-raschka.md) — 结构化书+notebook 路线，与 Zero to Hero 互补
- [Software 2.0（2017 博文）](https://karpathy.github.io/2018/09/06/software2/) — 学习即编程与端到端栈讨论
