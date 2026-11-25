---
type: concept
tags: [transformer, recurrent-depth, world-model, inference]
status: complete
updated: 2026-11-25
summary: "以共享权重迭代细化隐藏状态的循环深度：区分 FaceMind 已公开的 LoopWM、仅见于报道的通用循环模型命名，以及智谱独立公开的 GLM-5.4。"
related:
  - ../entities/paper-sa-2606-18208-looped-world-models.md
  - ../methods/generative-world-models.md
  - ../../roadmap/depth-wam.md
original:
  - ../../sources/papers/looped_world_models_2606_18208.md
  - ../../sources/sites/zai-glm-5-4-loop-transformer.md
sources:
  - ../../sources/papers/looped_world_models_2606_18208.md
  - ../../sources/sites/zai-glm-5-4-loop-transformer.md
---

# General Looped Model（通用循环模型）：循环深度与一手证据

## 一句话定义

**循环深度**让同一组权重在一次预测内多次更新隐藏状态，以额外计算换更深的处理，而不为每一轮另存一整组层权重。本页把 **General Looped Model（通用循环模型）** 作为检索入口，不将它标为已核实的单独论文、模型版本或产品，也不把它和智谱的 GLM 品牌画等号。

## 英文缩写速查

| 缩写 | 英文 | 本页含义 |
| --- | --- | --- |
| LoopWM | Looped World Models | FaceMind Research Asia 公开的循环世界模型论文 |
| KV | Key–Value | 注意力机制复用的键和值缓存 |
| GLM-5.4 | Z.ai GLM-5.4 | 智谱的模型名称，本文的独立语言模型对照 |

## 为什么重要

权重规模决定模型存储的一部分，实际执行轮次则影响算力、延迟和激活/缓存占用。对机器人规划有价值的问题不是「参数省多少」，而是在相同决策时限内，多做一次潜状态细化是否比多试一条候选轨迹更有效。

## 核心机制与流程

```mermaid
flowchart TD
    A["编码：观测与动作，或 token"] --> B["共享参数块：读取当前隐藏状态"]
    B --> C["写回细化后的隐藏状态"]
    C --> D{"达到退出条件或预算？"}
    D -->|"否"| B
    D -->|"是"| E["输出本次预测或交给后续层"]
    E --> F["按任务推进外层环境步或 token"]
    F --> A
```

图中内圈是**循环深度**，外圈才推进环境步或生成下一个 token。扩散采样的多轮去噪推进的是采样步，不能仅凭多次前向就推断每个去噪步内部也有共享深度循环。

| 一手来源 | 循环处理什么 | 退出或状态管理 | 证据边界 |
| --- | --- | --- | --- |
| FaceMind Research Asia 的 LoopWM | 同一环境步的动作条件潜状态 | 论文描述可变循环深度、门控早退；多步潜推演可延迟到末端解码 | ScienceWorld 和 ALFWorld 文本交互预测，不是真机验证 |
| Z.ai 的 GLM-5.4 | 自回归模型中间 28 层的 token 隐状态 | 两轮；逐 token 门控调节更新强度，第二轮共享部分 KV 并保留局部滑窗 KV | 30B 消融等参数但训练计算约多 10%；不是推理开销，也不是真机结果 |

## 工程实践

比较时分别记录参数数、每次预测的循环数、外层 rollout 长度、解码次数、成功率、P50/P95 延迟和峰值显存。机器人场景先测固定墙钟预算下的 rollout 质量，再比较固定循环、难度早退和候选动作批量大小；不要拿文本精确匹配率替代闭环成功率。

## 局限与核验边界

- 公开报道把「General Looped Model」用于串联 FaceMind 的 LoopWM 与内部的 Looped Diffusion。可核对的 FaceMind 一手材料是 LoopWM 论文；[FaceMind 官网世界模型预览](https://faceminda.com/world-model)也没有把该英文术语确认为正式模型名。
- 截至 2026-11-25，未找到可独立核对的 Looped Diffusion 官方全文、训练数据、代码或权重；不录入报道所列规模或性能数字。对 LoopWM 与 GLM-5.4 也不以流程图充当已验证的源码运行时序。
- 共享参数并不自动节省推理 FLOPs；少输出一些 token 也不保证更低端到端延迟。门控控制更新幅度，与早退减少循环次数是两件事。

## 关联页面

- [Looped World Models（循环世界模型）](../entities/paper-sa-2606-18208-looped-world-models.md)
- [Loop Behind：循环语言模型的后置循环解码](../entities/paper-loop-behind.md)：对 Huginn 式架构分别干预当前读出深度和历史 KV 深度；用于对比文本解码与世界模型内循环，不是同一个项目。
- [生成式世界模型](../methods/generative-world-models.md)
- [世界动作模型纵深路线](../../roadmap/depth-wam.md)

## 推荐继续阅读

先读 LoopWM 的 §3.2–3.5，弄清内循环、动作序列和末端解码；再看 Z.ai 博客的 Loop Architecture，比较固定双循环、门控更新与缓存共享三项设计各自解决的问题。

## 参考来源

- FaceMind Research Asia：[论文 HTML v1](https://arxiv.org/html/2606.18208v1) · [PDF v1](https://arxiv.org/pdf/2606.18208v1) · [一手通读归档](../../sources/papers/looped_world_models_2606_18208.md)
- Z.ai：[GLM-5.4 官方研究博客](https://z.ai/blog/glm-5.4) · [一手通读归档](../../sources/sites/zai-glm-5-4-loop-transformer.md)
- [报道中的命名线索](https://t.cj.sina.com.cn/articles/view/3996876140/ee3b7d6c0010186hc)（二手，仅供追溯词名，不作机制或性能证据）。
