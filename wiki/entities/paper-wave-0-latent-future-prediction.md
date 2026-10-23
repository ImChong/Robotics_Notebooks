---
type: entity
tags: [paper, vla, world-action-models, latent-future-prediction, lightweight-world-model]
status: complete
updated: 2026-10-22
project_id: wave-0
arxiv: "2610.18568"
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/wave-0_arxiv_2610_18568.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "WAVE-0（arXiv:2610.18568）：用 257M 参数的轻量潜在世界模型连接预训练 VLA 与世界动作模型，把 VLM 生成的潜在动作计划翻译为未来表征，避免完整视频生成的计算开销。"
---

# WAVE-0：用轻量潜在未来预测连接 VLA 与 WAM

**WAVE-0 在预训练 VLA 之上加一个小型潜在世界模型，让预测出的动作后果直接参与动作生成，而不生成完整视频。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界动作模型 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

摘要指出，多数 WAM 依赖昂贵的视频生成；而纯潜在方法要么只把预测未来当辅助目标，要么需要大规模视频预训练才能从次优的视觉转移中推断潜在动作。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.18568](https://arxiv.org/abs/2610.18568) |
| 作者 | Haomin Wang, Anqiong Ji, Ke Wang, Zhongshan Liu, Ru Ying, Ran Zheng, Zhipeng Zhang（arXiv 作者列表） |
| 机构 | 未核实（摘要页未列） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **底座：** 预训练 VLA，其 VLM 输出潜在动作计划。
- **轻量世界模型：** 257M 参数，把潜在动作计划翻译为预测的未来表征。
- **引导动作：** 预测的动作后果直接指导动作生成。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 评测与结果

以下数字来自 arXiv 摘要（论文 HTML 版未生成），本库未复现。

| 评测 | 论文报告成功率 |
|------|----------------|
| [RoboTwin 2.0](./robotwin.md) | 91.47% |
| [LIBERO](./libero-benchmark.md) | 98.7% |
| [LIBERO-Plus](./paper-rcl-2510-13626-libero-plus-in-depth-robustness-analysis-of-visi.md) / LIBERO-Pro / LIBERO-Para（分布偏移） | 88.0% / 61.4% / 75.0% |
| 真机任务 | 在 π0.5 上从 38.3% 提升到 73.3% |

摘要另指出世界模型直接用真实机器人动作、以反事实排序目标训练，不需要像素重建或大规模视频预训练。摘要未给出推理延迟。

## 与其他工作对比

| 路线 | 未来预测形式 | 与 WAVE-0 的差异 |
|------|--------------|------------------|
| [π0.5](./paper-pi05-open-world-vla.md)（真机底座） | 无显式未来预测 | WAVE-0 在其上加 257M 潜在世界模型，用预测后果条件化动作专家 |
| 视频生成式 WAM（见 [世界动作模型](../concepts/world-action-models.md)） | 生成未来视频帧 | 摘要称其计算昂贵；WAVE-0 只在潜在空间预测未来表征 |
| [FLARE](./paper-sa-2505-15659-flare-robot-learning-with-implicit-world-modelin.md) 等潜在未来辅助目标路线 | 预测未来潜表征作为辅助损失 | 摘要指出此类方法只把未来当辅助目标；WAVE-0 让预测直接输入动作生成 |
| [Motus](./paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md) 等潜在动作路线 | 从视频转移推断潜在动作 | 往往需大规模视频预训练；WAVE-0 直接在机器人动作上训练世界模型 |

除 π0.5 真机提升为论文同协议对照外，其余均为路线层面的定性区分，不共享评测口径。

## 结论

**一句话总判：用小型潜在世界模型提供未来信息，是在保留 WAM 收益的同时压低计算成本的一条路线，但收益需要与完整视频 WAM 对照。**

1. **未来预测可以留在潜在空间** — 摘要以避免视频生成开销为主要动机。
2. **世界模型规模可以很小** — 257M 参数是论文给出的规模。
3. **与 VLA 底座的衔接是关键** — 潜在动作计划的质量决定预测信息的可用性。

## 局限与风险

- 成功率为论文自报；摘要未给出推理延迟，轻量化收益尚缺时间开销数据。
- 对 VLM 潜在动作计划的依赖，使误差可能在两级之间累积（推测）。

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [世界动作模型](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [论文归档](../../sources/papers/wave-0_arxiv_2610_18568.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.18568](https://arxiv.org/abs/2610.18568)
