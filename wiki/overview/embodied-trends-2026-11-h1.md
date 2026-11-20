---
type: overview
tags:
  - vla
  - embodied-ai
status: draft
updated: 2026-11-20
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/duomo_embodied_2026_11_first_half.md
summary: "2026年11月上半月具身智能九篇论文的主题与技术介入点比较；不把作者206篇统计当作独立复算。"
---
# 2026 年 11 月上半月具身智能研究趋势：九篇技术地图

> 来源：多模空间公众号风向标（2026-11-20）。**206 篇**为原作者对其样本的统计；本文没有完整文章图表和去重样本，不能复核该数字。

## 一句话定义
将九篇论文按**技能规划、输入与表征、行为条件化、安全诊断、遮挡推演**整理为读论文地图，避免跨环境直接比较成功率。

## 英文缩写速查
| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| VLM | Vision-Language Model | 视觉语言骨干 |
| OOD | Out of Distribution | 分布外 |
| MCTS | Monte Carlo Tree Search | 树搜索 |

## 为什么重要
九篇论文研究问题不同：有些改训练时感知，有些改技能路由或任务层计划，还有些检查“文字解释与真实动作不一致”以及“看不见目标如何行动”。把它们放到统一任务榜单会误导。

## 按问题找论文
| 研究问题 | 官方论文版本 | 唯一知识节点 |
|---|---|---|
| NS-VLA | [2603.09542v4](https://arxiv.org/abs/2603.09542v4) | [查看](../entities/paper-ns-vla-2603-09542.md) |
| 3A-VLA | [2602.02212v2](https://arxiv.org/abs/2602.02212v2) | [查看](../entities/paper-3a-vla-2602-02212.md) |
| PRISM | [2611.08917v1](https://arxiv.org/abs/2611.08917v1) | [查看](../entities/paper-prism-2611-08917.md) |
| AnyCamVLA | [2603.05868v3](https://arxiv.org/abs/2603.05868v3) | [查看](../entities/paper-anycam-vla.md) |
| A³-VLA | [2611.06447v2](https://arxiv.org/abs/2611.06447v2) | [查看](../entities/paper-a3-vla-attention-alignment.md) |
| DynaFLIP | [2605.30350v2](https://arxiv.org/abs/2605.30350v2) | [查看](../entities/paper-dynaflip-2605-30350.md) |
| PersonaDrive | [2606.12616v2](https://arxiv.org/abs/2606.12616v2) | [查看](../entities/paper-personadrive-2606-12616.md) |
| Say the Same, Act Differently | [2611.04166v1](https://arxiv.org/abs/2611.04166v1) | [查看](../entities/paper-say-the-same-act-differently-2611-04166.md) |
| SCOUT | [2611.01600v1](https://arxiv.org/abs/2611.01600v1) | [查看](../entities/paper-scout-2611-01600.md) |

## 核心机制
```mermaid
flowchart TD
  R["具身任务中的泛化瓶颈"] --> P["规划和技能"]
  R --> V["视觉与动力学"]
  R --> D["驾驶和安全"]
  R --> O["遮挡场景"]
  P --> N["NS-VLA / 3A-VLA / PRISM"]
  V --> A["AnyCamVLA / A³-VLA / DynaFLIP"]
  D --> S["PersonaDrive / Say the Same, Act Differently"]
  O --> T["SCOUT"]
```
这是便于阅读的**知识分类图**，不是九篇论文共同使用的训练框架。

## 工程实践
- **已找到官方可运行代码入口：** [NS-VLA](../entities/paper-ns-vla-2603-09542.md)、[DynaFLIP](../entities/paper-dynaflip-2605-30350.md)、[PersonaDrive](../entities/paper-personadrive-2606-12616.md)；附各自实际入口的静态 Mermaid 调用时序图。
- **已有节点复用：** [AnyCamVLA](../entities/paper-anycam-vla.md)、[A³-VLA](../entities/paper-a3-vla-attention-alignment.md) 已在仓库，**本次不重复创建**；既有节点记录的全文版本与本次用户列示的 v3/v2 不完全一致，后续需专题审校。
- 入口：[VLA 纵深路线 Stage 5](../../roadmap/depth-vla.md)；优先按自己关心的任务与模块选读。

## 局限与风险
- **206 篇、分类图、真机品牌图**来自公众号描述而非可复算原始表格；没有把它们当作可验证统计结果。
- 三维游戏、CARLA 驾驶、桌面操作、真机货架与对抗诊断不是同一评测任务，成功率和可靠性不能横比。
- 未逐篇完成原论文所有附录与结果复核；节点保留 draft 并明确标注证据缺口。

## 选读路径
### 规划、技能与动作抽象
[NS-VLA](../entities/paper-ns-vla-2603-09542.md) → [3A-VLA](../entities/paper-3a-vla-2602-02212.md) → [PRISM](../entities/paper-prism-2611-08917.md)。区别神经符号指针、游戏语义抽象与技能适配器路由。

### 视觉感知与动态表征
[AnyCamVLA](../entities/paper-anycam-vla.md) → [A³-VLA](../entities/paper-a3-vla-attention-alignment.md) → [DynaFLIP](../entities/paper-dynaflip-2605-30350.md)。依次对照测试期视角合成、训练期注意力监督与动力学三模态表示。

### 驾驶、行为诊断与探索
[PersonaDrive](../entities/paper-personadrive-2606-12616.md) → [Say the Same, Act Differently](../entities/paper-say-the-same-act-differently-2611-04166.md) → [SCOUT](../entities/paper-scout-2611-01600.md)。三篇分别属于风格条件化、语言行为偏移诊断与遮挡下规划，任务并不一致。

## 关联页面
- [VLA 方法总览](../methods/vla.md)
- [机器人操作](../tasks/manipulation.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)

## 参考来源
- [多模空间公众号资料索引](../../sources/blogs/duomo_embodied_2026_11_first_half.md)
- [公众号原始链接](https://mp.weixin.qq.com/s/v-PpSCF-c7qHQXzlSYXyMw)
- 九篇原始 arXiv 版本及节点上方表格逐一列出；公众号原图及全部论文附录**未获完整读取**。

## 推荐继续阅读
- [VLA](../methods/vla.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
