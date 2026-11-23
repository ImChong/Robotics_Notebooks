---
type: entity
project_id: prenav-2611-15093
tags: [paper, navigation, diffusion, object-search]
status: complete
updated: 2026-11-21
arxiv: "2611.15093"
original:
  - ../../sources/papers/prenav-2611-15093.md
sources:
  - ../../sources/papers/prenav-2611-15093.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "在半静态室内布局中先生成全局搜索优先顺序，减少寻找目标时的多次语义决策。"
---

# Decide Before You Move: Diffusion-Generated Search Reference for Zero-Shot Open-Vocabulary Object Navigation in Semi-Static Environments｜PreNav：出发前生成全局搜索路线的开放词汇导航

## 一句话定义

**PreNav：出发前生成全局搜索路线的开放词汇导航**：在半静态室内布局中先生成全局搜索优先顺序，减少寻找目标时的多次语义决策。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| OVON | Open-Vocabulary Object Navigation | 开放词汇目标导航 |
| RGB | Red Green Blue | 彩色视觉 |
| DDPM | Denoising Diffusion Probabilistic Models | 扩散生成模型 |

## 为什么重要

在半静态室内布局中先生成全局搜索优先顺序，减少寻找目标时的多次语义决策。 需要结合任务边界、数据/算力条件及作者真实实现来判断，而不是只看展示视频。

## 核心信息

| 证据项 | 官方信息 |
|---|---|
| 论文 | [arXiv:2611.15093v1](https://arxiv.org/abs/2611.15093) |
| 源码 | 无经核实的官方可运行完整发布 |

## 核心原理

从墙体/大型家具语义建立目标类别的支撑概率图，把墙体栅格、起始位置和目标先验送入扩散模型，输出带搜索宽度的全局搜索路径；在线局部规划沿参考路线观察候选位置。

### 流程总览

```mermaid
flowchart LR
  V0["墙体与大型家具地图"]
  V1["目标类别空间先验"]
  V2["扩散生成全局搜索参考"]
  V3["局部路径跟踪与观察"]
  V4["目标确认及终止"]
  V0 --> V1
  V1 --> V2
  V2 --> V3
  V3 --> V4
```

该图为研究问题与方法/控制流，实际软件调用须看源码时序图及开放边界。

## 源码运行时序图

**不适用：** 官方源码尚无经过核实的可运行完整入口，或当前公开的代码不具备端到端可执行部署能力，故不虚构 sequenceDiagram。

## 工程实践

1. 检查输入观察、动作/轨迹表示、数据来源和实验任务的部署条件是否一致。
2. 等待作者公开或确认源码后再写运行时序。
3. 将仿真指标、真机成功率、推理延迟与安全/碰撞检查分开汇报。

## 实验与评测

基于 ScanNet++ 75/8 个训练/验证室内场景构造搜索路线监督，并在其他仿真与少量真实导航场景对照。

## 结论

**在半静态室内布局中先生成全局搜索优先顺序，减少寻找目标时的多次语义决策。**

1. 先复核指标口径与论文约束，再比较其他策略或系统。
2. 将作者验证的模块和尚未发布/未复现的模块分开记录。
3. 把误差、失败模式与传感器/真实场景分布偏移作为后续验收依据。

## 与其他工作对比

- **对比 `SGM` 语义地图补全导航基线（Gibson 6 类 600 episodes & 10 类 1,000 episodes，Tables I–II）：** 在 6 类子集上，无先验 `SGM` 仅达 55.0% SR / 29.6% SPL；即使向 `SGM` 同步输入墙体与同范围 5 类家具的像素级真值语义图先验，其表现升至 62.0% SR / 36.2% SPL，而仅需每件家具 2D 包围盒的 **PreNav** 在相同家具先验下取得 **64.3% SR / 39.2% SPL**（+2.3% SR / +3.0% SPL）；若再补齐场景中未标注的其余家具 2D 框（`SGM` 类别体系不支持），PreNav 进一步跃升至 **70.3% SR / 39.8% SPL**。在 10 类全集上，同先验下 PreNav 仍以 +1.4% SR 领先 `SGM`（SPL 持平）。
- **对比感知前端与标注口径（Table III）：** 对照 `Mask R-CNN`、`SAM 3` 与 `3DSG oracle`，表明 `book/bottle/clock/cup` 四类扩展目标的瓶颈主要在感知与地图标注分歧（例如不同标注源缺失类别），而非全局搜索路线生成本身。

## 局限与风险

依赖保持相对稳定的墙体及家具布局，无法直接适用于完全未知或频繁搬动物体的大变动环境；匿名代码网站无法可靠读取源码入口。

## 关联页面

- [29 篇具身论文研究地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder 操作任务难度分层](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [论文原文归档](../../sources/papers/prenav-2611-15093.md)

## 推荐继续阅读

- [原始论文 HTML](https://arxiv.org/html/2611.15093v1)
- [原文 PDF](https://arxiv.org/pdf/2611.15093)
