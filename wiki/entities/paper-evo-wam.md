---
type: entity
tags:
  - paper
  - world-action-model
  - video-world-model
  - self-improvement
  - robot-learning
status: complete
updated: 2026-10-22
project_id: evo-wam
arxiv: "2609.38057"
project: https://evo-wam.github.io/
code: https://github.com/Clausy9/EVO-WAM
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../overview/hub-embodied-foundation-model.md
sources:
  - ../../sources/papers/evo_wam_arxiv_2609_38057.md
summary: "EVO-WAM（2026）：让世界动作模型通过视频-动作验证的已完成前缀迭代自我重训，无需新示范即可适配新任务；RoboTwin 2.0 与真机长时程任务均有提升。"
---

# EVO-WAM（通过视频-动作验证演化世界动作模型）

## 一句话定义

**EVO-WAM 让世界动作模型在自己的 rollout 中筛选出真正完成任务且视频与动作一致的前缀，再用这些前缀迭代重训，从而无需新示范地适配未见任务。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 同时预测未来视频与动作的模型 |
| VLM | Vision-Language Model | 用于挑选任务完成前缀的判别器 |
| IDM | Inverse Dynamics Model | 从视频推断动作，检验视频-动作一致性 |
| RoboTwin 2.0 | RoboTwin 2.0 | 论文使用的双臂仿真基准 |

## 为什么重要

- WAM 适配新任务通常需要新的专家示范或外部执行；本文尝试只用模型自身 rollout 完成适配。
- 验证环节（VLM 选择 + IDM 一致性）决定了哪些自生成数据被保留，是方法的核心。

## 方法与机制

- 加入状态预测与多帧上下文，使模型能完成完整的自回归 rollout。
- 用 VLM 选出完成任务的前缀，并用 IDM 检验视频与动作的一致性。
- 只保留经验证的前缀，迭代地在其上重训练。

## 实验与评测

- RoboTwin 2.0 的 7 个任务：Cosmos3 26.9% → 68.0%，DreamZero 28.5% → 46.4%（据摘要）。
- 真机 3 个长时程任务：Cosmos3 20.0% → 76.7%（据摘要）。

## 与其他工作对比

- 摘要未给出与专家示范微调的直接对比数值；论文以「无需新示范」为卖点。

## 结论

本文的核心是把「自生成数据的验证」作为世界动作模型自我改进的关键环节。

- 自生成数据必须经过验证才能用于重训，否则会放大错误。
- 视觉判断（VLM）与动作一致性检验（IDM）共同过滤数据。
- 证据主要来自摘要与两组任务，尚未核实在其他基础模型上的泛化。

## 工程实践

- 官方代码：`Clausy9/EVO-WAM` 仓库目前仅含 README、citation.bib 与 demo 素材。项目侧 news 写明官方代码「pending approval, release date TBD」，**未发布**。
- 第三方复现：`XiaohanLei/open-evo-wam`（非官方，未核实）。
- 权重与数据：仓库中未见（已核实）。

## 局限与风险

- 官方代码未发布，复现依赖第三方实现或论文细节。
- 推测：VLM 与 IDM 的判断误差会直接影响保留的训练数据，需要检查过滤精度。
- 推测：迭代自训练可能使模型分布收窄，需防止过拟合已成功的前缀。

## 关联页面

- [World Action Models](../concepts/world-action-models.md) — 同时预测视频与动作的模型背景
- [Generative World Models](../methods/generative-world-models.md) — 视频世界模型方法族

## 参考来源

- [EVO-WAM 论文来源归档](../../sources/papers/evo_wam_arxiv_2609_38057.md)

## 推荐继续阅读

- [arXiv 2609.38057](https://arxiv.org/abs/2609.38057)
- [项目页](https://evo-wam.github.io/)
- [Hugging Face Daily Paper](https://huggingface.co/papers/2609.38057)
