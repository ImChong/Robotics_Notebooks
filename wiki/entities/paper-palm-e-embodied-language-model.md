---
type: entity
tags: ["paper", "vla", "multimodal", "foundation-model", "google", "hmi-papers"]
status: complete
updated: 2026-07-31
arxiv: "2303.03378"
venue: "HMI curated · 2023"
summary: "PaLM-E（HMI P053）：把连续相机与机器人状态投影成与文本相同的嵌入序列，使视觉、状态与语言共享自回归推理上下文（输出仍主要在语言层）。"
related:
  - ../concepts/foundation-policy.md
  - ../methods/vla.md
  - ../methods/robotics-transformer-rt-series.md
  - ./openvla.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p053_palm-e-embodied-language-model.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# PaLM-E（HMI P053）

**PaLM-E**（*PaLM-E: An Embodied Multimodal Language Model*，2023，[arXiv:2303.03378](https://arxiv.org/abs/2303.03378)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P053**，主分类为 **世界模型、VLA与Agent**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

把连续相机与机器人状态投影成与文本相同的嵌入序列，使视觉、状态与语言共享自回归推理上下文（输出仍主要在语言层）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉语言模型前驱形态 |
| VLA | Vision-Language-Action | 后续直接输出动作的路线 |
| LLM | Large Language Model | 语言主干 |
| Embodied | Embodied AI | 把传感器状态接入语言模型推理 |

## 为什么重要

- 图像可由ViT等视觉编码器处理，机器人状态或3D信息也由对应编码器转成与词向量同维的嵌入。这些连续嵌入可以出现在文字指令之前、之后或中间，然后与预训练PaLM一起端到端微调。训练目标仍是预测文字token，所以跨模态迁移发生在LLM的共享表示中：视觉问答、图像描述和语言任务的数据，可以改善机器人场景的概念理解。
- 在 HMI 六条技术路线中属于 **世界模型、VLA与Agent**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P053 |
| 年份 | 2023 |
| 分组 | 世界模型、VLA与Agent |
| 开源状态 | 未开源模型权重（Google 研究发布） |
| 原文 | https://arxiv.org/abs/2303.03378 |

## 核心原理

PaLM-E经常被简写成“多模态大模型做机器人”，但它真正关键的设计是输入接口：不把图像转成一句固定文字，也不为机器人单独搭一套语义网络，而是用可学习编码器把图像、3D感知和本体状态映射到LLM词嵌入空间，与文字交错排成一条“多模态句子”。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["PaLM-E"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

机器人训练样本把当前多模态观测、任务文本和期望计划/答案排成序列，统一用下一token目标优化。模型内部状态更适合表示物体关系、步骤和语义记忆，而不是精确接触动力学。它可以在新图像到来后重新生成后续文本计划，但每轮是否任务成功、物体是否抓稳以及应该调用哪个可执行技能，都要由系统提供可观察反馈。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 未开源模型权重（Google 研究发布） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（未开源模型权重（Google 研究发布））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**PaLM-E 应作为 HMI「世界模型、VLA与Agent」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：未开源模型权重（Google 研究发布）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 在机器人任务中，PaLM-E生成的是计划步骤、问题答案或可交给下游策略的文本命令。它还能在执行过程中继续接收新图像，根据环境变化重新规划。但从文本计划到机械臂轨迹之间，仍然需要技能策略、成功检测和底层控制。因此把PaLM-E当成后来意义上直接产生连续动作的VLA，会把系统层级搞错。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（PaLM-E） | [VLA](../methods/vla.md) | [RT 系列](../methods/robotics-transformer-rt-series.md) | [OpenVLA](openvla.md) |
|------|------------------|--------------------------|--------------------------------------------------------|-----------------------|
| 输出层 | 语言层：计划步骤/答案文本 | 直接输出低层动作 | 动作 token | 离散动作 token |
| 输入接口 | 图像/3D/状态投影进 LLM 词嵌入，交错成多模态句子 | 视觉+语言条件→动作 | 图像+指令→动作 | VLM 骨干→动作 |
| 训练目标 | 下一 token（语言）预测 | 行为克隆/动作监督 | 动作监督 | 动作监督 |
| 关系/取舍 | 需下游技能策略+成功检测执行，勿当直接产动作的 VLA | PaLM-E 之后的范式抽象 | 可作 PaLM-E 计划的执行层 | 端到端产动作，去掉语言中转 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [foundation-policy](../concepts/foundation-policy.md)
- [vla](../methods/vla.md)
- [robotics-transformer-rt-series](../methods/robotics-transformer-rt-series.md)
- [openvla](./openvla.md)

## 参考来源

- [sources/papers/hmi_p053_palm-e-embodied-language-model.md](../../sources/papers/hmi_p053_palm-e-embodied-language-model.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [arXiv:2303.03378](https://arxiv.org/abs/2303.03378)
- [HMI 逐篇解读 P053](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P053.md)
