---
type: entity
tags:
  - paper
  - vla
  - latency
  - dynamic-manipulation
  - action-chunking
  - sjtu
status: complete
updated: 2026-08-17
arxiv: "2608.14379"
related:
  - ../methods/vla.md
  - ../methods/action-chunking.md
  - ../concepts/embodied-fm-latency-generalization-tradeoff.md
  - ../tasks/manipulation.md
  - ./paper-dypes-vla.md
  - ./paper-gsr-paravla.md
  - ../queries/vla-deployment-guide.md
sources:
  - ../../sources/papers/reflexvla_arxiv_2608_14379.md
  - ../../sources/sites/reflexvla-github-io.md
summary: "ReflexVLA（arXiv:2608.14379，交大）：ReflexBench 六任务延迟感知评测 + 1B VLA（冻结 DINOv3 未来预测、视觉骨干时序融合、CUDA Graph）；均值 50.4%、LIBERO 97.2%；代码录用后开放。"
---

# ReflexVLA：动态任务低延迟 VLA

**ReflexVLA**（*Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation*，[arXiv:2608.14379](https://arxiv.org/abs/2608.14379)，[项目页](https://reflexvla.github.io/)）由 **上海交通大学（SJTU）** 提出：先建延迟感知基准 **ReflexBench**（仿真不因推理而暂停世界），再在 VLA-Adapter 骨干上加未来隐特征预测、视觉中间层时序融合与系统级加速，做反应关键操纵。

## 一句话定义

**动态操纵的瓶颈不只是「会不会做」，而是「想得够不够早、算得够不够快」——ReflexBench 把延迟写进评测，ReflexVLA 用预测+时序+CUDA Graph 在 1B 上同时抬成功率、压延迟。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RTF | Real-Time Factor | 仿真时间 / 墙钟时间，用来把真机延迟注入仿真 |
| MHA | Multi-Head Attention | 本文在视觉中间层做因果时序融合 |
| CUDA Graph | CUDA Graph replay | 固定计算图一次捕获、每步回放以降调度开销 |
| LIBERO | Lifelong Robot Learning benchmark | 静态操纵对照榜 |

## 为什么重要

- **静态榜漏掉反应任务：** 接球、传送带、旋转插孔要求在物体还在动时出动作。
- **多数仿真「暂停世界」：** 推理时环境冻结，掩盖感知–执行错位；本基准显式注入同步/异步延迟。
- **预测与加速常被拆开做：** 未来推理往往更慢，加速工作又不管提前量；本文把两者绑在同一 1B 配方。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 上海交通大学（SJTU） |
| **骨干** | VLA-Adapter 族：DINOv2+SigLIP 224、Qwen2.5-0.5B、连续回归头；约 **1B** |
| **基准** | ReflexBench 六任务；对照 LIBERO；真机 AgileX Piper |
| **开源** | **宣称录用后开源**（项目页 Code After acceptance；截至 2026-08-17 无仓） |

## 核心原理

### 方法栈

| 模块 | 机制 |
|------|------|
| 未来预测 | \(H\) 个 future token 对齐 action chunk；目标为冻结 DINOv3 特征；masked cosine；\(\lambda=0.05\) |
| 时序融合 | 多视角×多历史在视觉中间层因果 MHA；语言侧仍只吃当前帧 token |
| 延迟优化 | 全部 view-frame 一次 batched 编码 + 整推理图 CUDA Graph |
| 部署协议 | 异步 chunk=8、执行 horizon=2（消融后的默认） |

### 流程总览

```mermaid
flowchart TB
  obs["多视角 + 短历史 RGB"] --> vit["batched ViT"]
  vit --> fuse["中间层因果时序融合"]
  fuse --> lm["Qwen2.5-0.5B + action query"]
  lm --> act["动作 chunk"]
  lm --> fut["future token → 冻结 DINOv3 空间"]
  graph["CUDA Graph replay"] --> act
```

可训练未来目标会把表示空间一起拧歪（传送带 SR 36.8→**4.9**）；冻结目标则 36.8→**62.8**。

## 源码运行时序图

**不适用（官方可运行代码尚未发布）。** 截至 2026-08-17：项目页标注 **Code After acceptance**。发布后应补：数据加载 → 未来/动作联合训练 → 异步 chunk + CUDA Graph 推理的 `sequenceDiagram`。

## 工程实践

| 项 | 建议 / 论文设定 |
|----|----------------|
| 何时用 | 物体在动、错过时间窗即失败；静态家务不是主战场 |
| 训练 | 六任务各 200 demo 共训单策略；2 帧历史 |
| 评测 | 仿真与控制解耦；可用墙钟延迟 / RTF 注入 |
| 加速 | 架构不变，只减框架与 kernel 调度；融合后 125 ms → **65.0 ms** |
| 复现现状 | **等官方代码**；先读延迟协议与消融表做选型 |

## 实验与评测

| 设定 | ReflexVLA | 读点 |
|------|-----------|------|
| ReflexBench 均值 | **50.4%** | 打平 PUMA 50.2%（4B），远超骨干 30.3% 与 \(\pi_{0.5}\) 36.9% |
| 传送带 | **73.8%** | 相对骨干 36.8 的主增益场 |
| 接球 / 旋转插销 | 7.3% / 12.4% | 仍难；大模型也低 |
| LIBERO | **97.2%** | 与 VLA-Adapter 97.3 持平，动态模块未牺牲静态榜 |
| 真机 Piper | 16/20、22.5 键/30s、6.7/10 球 | 优于 SmolVLA 与 PUMA |

Q1 协议：低频时异步更吃亏；高频异步 + 大 chunk + 短 horizon 最好。

## 结论

**ReflexVLA 的可迁移主张是「把未来写进表示、把历史留在视觉、把延迟交给系统」：1B 就能在反应任务上追上 4B 动态专精，同时保住 LIBERO。**

1. **真影响：评测必须带延迟** — 暂停世界的仿真会高估动态 VLA。
2. **真影响：冻结未来目标** — 可训练 DINOv3 头会崩；稳定语义空间是前提。
3. **真影响：中间层融合** — 运动线索在中层，不在最语义的末层。
4. **真影响：65 ms 也改 SR** — 加速不是纯工程甜点，反应任务上直接换成功率。
5. **次要代价：未大规模预训练、未试 RTC** — 上限可能还没打满。
6. **部署读法：** 代码未发；先按异步 chunk=8 / horizon=2 对照自有动态任务。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| DynamicVLA / PUMA | 同属动态操纵；本文把延迟协议与 1B 系统加速写进同一套 |
| [Action Chunking](../methods/action-chunking.md) | 通用缓冲机制；本文用实验钉死「大 chunk + 短执行地平线」 |
| [实时性↔泛化](../concepts/embodied-fm-latency-generalization-tradeoff.md) | 概念页讲带宽墙；本文是 1B 侧的破墙实例 |
| [DyPES-VLA](./paper-dypes-vla.md) | 未来预测塑跨本体先验；本文未来预测服务反应提前量 |
| 大通才 \(\pi_{0.5}\) / OpenVLA-OFT | 静态/开放词汇强，ReflexBench 均值明显落后 |

## 局限与风险

- **开源未落地：** 无法复核融合层、Graph 捕获范围与采数规划器。
- **任务仍偏玩具动态：** 六仿真 + 三真机，不是长程家务。
- **接球等任务全体低分：** 「反应关键」远未解决。
- **微调-only 模块：** 作者承认未进大规模预训练。

## 关联页面

- [VLA](../methods/vla.md) — 方法母页
- [Action Chunking](../methods/action-chunking.md) — 异步 chunk / horizon 协议
- [具身大模型实时性↔泛化取舍](../concepts/embodied-fm-latency-generalization-tradeoff.md) — 延迟墙
- [Manipulation](../tasks/manipulation.md) — 操作任务背景
- [DyPES-VLA](./paper-dypes-vla.md) — 另一条「未来预测进 VLA」
- [GSR / ParaVLA](./paper-gsr-paravla.md) — 同校 VLA-Adapter / SmolVLA 骨干对照
- [VLA 真机部署指南](../queries/vla-deployment-guide.md) — 异步执行

## 参考来源

- [reflexvla_arxiv_2608_14379.md](../../sources/papers/reflexvla_arxiv_2608_14379.md) — 论文摘录与开源核查
- [reflexvla-github-io.md](../../sources/sites/reflexvla-github-io.md) — 项目页核查
- [arXiv:2608.14379](https://arxiv.org/abs/2608.14379) — 原文

## 推荐继续阅读

- [ReflexVLA 项目页](https://reflexvla.github.io/)
- [VLA-Adapter](https://arxiv.org/abs/2509.09372) — 骨干论文
- [DynamicVLA](https://arxiv.org/abs/2601.22153) — 动态物体 VLA 对照
