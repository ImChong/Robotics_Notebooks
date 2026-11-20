---
type: entity
tags: [paper, transformer, attention, research]
status: draft
updated: 2026-11-20
project_id: attention20-2103-03206-perceiver
arxiv: "2103.03206"
summary: "Perceiver：利用迭代注意力进行通用感知：通过输入到固定大小 latent array 的非对称交叉注意力，把高维多模态输入压缩为潜变量，再在潜变量间迭代自注意力；避免在全部输入位置间执行昂贵的全局自注意力。"
related:
  - ../comparisons/attention-architectures-robotics-20-papers.md
  - ../methods/bc-with-transformer.md
  - ../methods/vla.md
  - ./paper-attention-is-all-you-need.md
  - ./paper-act.md
original:
  - ../../sources/papers/attention20_2103_03206.md
sources:
  - ../../sources/papers/attention20_2103_03206.md
---

# Perceiver: General Perception with Iterative Attention（Perceiver：利用迭代注意力进行通用感知）

## 一句话定义

通过输入到固定大小 latent array 的非对称交叉注意力，把高维多模态输入压缩为潜变量，再在潜变量间迭代自注意力；避免在全部输入位置间执行昂贵的全局自注意力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| CA | Cross-Attention | 输入到潜变量的非对称交互 |
| SA | Self-Attention | 潜变量之间的关联建模 |
| FFN | Feed-Forward Network | 注意力层之后的前馈变换 |

## 为什么重要

- **解决的问题：** 通过输入到固定大小 latent array 的非对称交叉注意力，把高维多模态输入压缩为潜变量，再在潜变量间迭代自注意力；避免在全部输入位置间执行昂贵的全局自注意力。
- **机器人关联（跨论文迁移判断，非本文直接验证）：** 为机器人多相机与本体状态的异构 token 压缩提供架构参考，实际部署需核对输入采样和延迟。

## 核心原理

通过输入到固定大小 latent array 的非对称交叉注意力，把高维多模态输入压缩为潜变量，再在潜变量间迭代自注意力；避免在全部输入位置间执行昂贵的全局自注意力。

## 流程总览

```mermaid
flowchart LR
  A["多模态高维输入"] --> B["cross-attention 写入 latent"] --> C["latent self-attention 迭代"] --> D["任务输出"]
```

**图示范围：** 基于论文摘要和已公开方法主线的高层数据流；具体模块实现细节及源码调用关系待全文和代码核验，不应当作复现级结构图。

## 实验与评测

- 论文展示图像、音频、视频与点云等跨模态实验，重点是输入规模与任务适应能力；不能直接推导为机器人操作成功率。
- **读数边界：** 不将该论文的识别/检测/操控指标跨数据集直接比较；实验表格、随机种子与消融设置仍需全文核验。

## 结论

**这篇论文的主要价值在于注意力结构和任务约束之间的工程取舍，尚不能由摘要证实真实机器人部署收益。**

1. **结构重点** — 通过输入到固定大小 latent array 的非对称交叉注意力，把高维多模态输入压缩为潜变量，再在潜变量间迭代自注意力
2. **实证范围** — 论文展示图像、音频、视频与点云等跨模态实验，重点是输入规模与任务适应能力
3. **部署前确认** — 固定潜变量容量可能成为信息瓶颈；多模态实验不等于真实机器人闭环验证。

## 与其他工作对比

- [Transformer 基础](./paper-attention-is-all-you-need.md) 提供全局注意力基线，本文体现其在特定任务或计算约束下的变体/评估。
- [20 篇注意力架构与机器人策略对照](../comparisons/attention-architectures-robotics-20-papers.md) 按视频、三维、机器人策略与计算优化归类。

## 工程实践

- 为机器人多相机与本体状态的异构 token 压缩提供架构参考，实际部署需核对输入采样和延迟。
- 选择实现时需核实官方仓库、输入输出形状、训练数据和硬件运行条件，不应根据摘要推断可运行入口。

## 源码运行时序图

**暂不绘制：** 官方运行入口与实际源码调用关系尚未核实。此处不使用假想的通用 `sequenceDiagram` 冒充源码分析；待核验训练/推理入口后补充。

## 局限与风险

- 固定潜变量容量可能成为信息瓶颈；多模态实验不等于真实机器人闭环验证。
- **资料状态：** 当前只完成题录、摘要与方法主线核对，**尚未完成全文及附录通读**；本页按草稿收录，不标注健康度满分。

## 相关方法与任务

以下链接是**应用关联**，不是本文已在机器人场景验证这些工程收益：

- [bc-with-transformer](../methods/bc-with-transformer.md) — 异构观测压缩可作为 Transformer 模仿学习输入前端
- [vla](../methods/vla.md) — 视觉、语言、机器人本体 token 的融合可参考潜变量交叉注意力

## 关联页面

- [Attention Is All You Need](./paper-attention-is-all-you-need.md)
- [ACT 机器人动作分块](./paper-act.md)
- [20 篇注意力相关论文对照](../comparisons/attention-architectures-robotics-20-papers.md)

## 参考来源

- [本论文原始来源归档](../../sources/papers/attention20_2103_03206.md)
- [一手论文](https://arxiv.org/abs/2103.03206)

## 推荐继续阅读

- [原始论文/会议出版页](https://arxiv.org/abs/2103.03206)
- [Transformer 基础](./paper-attention-is-all-you-need.md)
