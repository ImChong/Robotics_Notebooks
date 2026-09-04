---
type: overview
tags: [hub, hub-cross-embodiment, transfer, any2any, retargeting]
status: complete
updated: 2026-09-04
related:
  - ../concepts/embodied-foundation-model-hardware-codesign.md
  - ../queries/cross-embodiment-transfer-strategy.md
  - ../entities/paper-any2any-cross-embodiment-wbt.md
  - ../entities/paper-emergent-transfer-cross-config.md
  - ../entities/paper-last-hd-latent-physical-reasoning.md
  - ../entities/generalist-gen15-one-shot.md
  - ../entities/paper-host-one-shot-human-video.md
  - ../entities/generalist-gen1-thousand-hands.md
  - ../entities/gemini-robotics.md
  - ../entities/paper-xmorph.md
  - ../entities/paper-ucag-p.md
  - ../entities/paper-crosstracer.md
summary: "跨具身迁移知识链汇总：不同机器人形态、仿真与真机之间的策略/动作迁移，重定向、域随机与 Any2Any 类方法的选型与失败模式；含同形态跨配置遗留数据三相迁移（Emergent Transfer）、末端/工具接口多样性（如 GEN-1 千手）、physical prompting 人→机/sim→真机（GEN-1.5）、人体→非人形腿式行为先验（X-Morph）与闭源 On-Device 快速适配（Gemini Robotics 2）轴。"
---

# 跨具身迁移（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=imitation-learning) 与 [路线页](../../roadmap/depth-imitation-learning.md)。

## 一句话定义

**跨具身迁移** 研究如何把在 **某一机器人形态、仿真环境或数据源** 上学到的技能，迁移到 **不同骨架、尺寸或硬件平台**，而不完全重训。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Embodiment | Robot Embodiment | 机器人形态/硬件实例 |
| Transfer | Cross-Embodiment Transfer | 跨形态技能迁移 |
| Retargeting | Motion Retargeting | 跨骨架动作映射（迁移前置） |
| DR | Domain Randomization | 扩宽训练分布以提升迁移 |
| OXE | Open X-Embodiment | 跨具身开源数据倡议 |

## 为什么重要

- **数据与硬件碎片化**：不可能每个形态都从零采集全套示范。
- **Sim2Real 的姊妹问题**：不仅是 sim→real，还有 human→robot、大→小人形。
- **WBT 与 VLA 共同痛点**：参考动作与策略接口需对齐目标机体。
- **与「自研本体」区分**：跨具身迁移研究 **有限接口下的技能搬运**；不等于「模型可脱离本体定义」——见 [具身大模型与本体协同设计](../concepts/embodied-foundation-model-hardware-codesign.md)。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| Query | 迁移策略决策树 | [Cross-Embodiment Transfer Strategy](../queries/cross-embodiment-transfer-strategy.md) |
| 概念 | 重定向与迁移 | [Motion Retargeting](../concepts/motion-retargeting.md) |
| 实体 | Any2Any WBT | [Any2Any Cross-Embodiment WBT](../entities/paper-any2any-cross-embodiment-wbt.md) |
| 实体 | Emergent Transfer（同形态跨配置） | [Emergent Transfer](../entities/paper-emergent-transfer-cross-config.md) — 遗留示教在 τ(T) 前无效、越过后陡升 |
| 实体 | LaST-HD 人手→机器人 VLA | [LaST-HD](../entities/paper-last-hd-latent-physical-reasoning.md) |
| 实体 | GEN-1.5 physical prompting | [GEN-1.5](../entities/generalist-gen15-one-shot.md) — 闭源；仿真/人示范作 prompt 驱动真机 |
| 实体 | HOST 人→机 one-shot | [HOST](../entities/paper-host-one-shot-human-video.md) — 开源；进度流形 + 自接地；双臂 ARX |
| 实体 | GEN-1 千手（跨末端/工具） | [GEN-1 千手](../entities/generalist-gen1-thousand-hands.md) — 闭源产业样本；同一基座跨 ~9k 末端变体 |
| 实体 | Gemini Robotics 2 On-Device | [Gemini Robotics](../entities/gemini-robotics.md) — 闭源；博客称新双臂本体数小时 / &lt;200 例适配 |
| 实体 | X-Morph 人体→非人形腿式 | [X-Morph](../entities/paper-xmorph.md) — 重定向+物理校正+跟踪蒸馏；Go2/六足/B2-Z1 |
| 实体 | UCAG-P 相机系锚点 VLA | [UCAG-P](../entities/paper-ucag-p.md) — 共享腕/抓取几何，翻译器出各本体命令；操作通才而非 WBT |
| 实体 | CrossTracer 导航像素残差 | [CrossTracer](../entities/paper-crosstracer.md) — 同一语义轨迹按轮式/腿式改可通行路径；**不是** WBT 换骨架 |
| 概念 | 角色动画 vs 机器人 | [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md) |
| 数据 | 跨具身数据集 | [Open X-Embodiment](../concepts/open-x-embodiment.md) |

## 与其他知识链的关系

- **[动作重定向](./hub-motion-retargeting.md)**：跨骨架动作对齐。
- **[WBT](./hub-wbt.md)**：跟踪策略的跨形态扩展。
- **[Sim2Real](./hub-sim2real.md)**：仿真-真机是跨具身特例。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Emergent Transfer：跨配置遗留数据何时开始有用](../entities/paper-emergent-transfer-cross-config.md) — 同形态换相机/夹爪后的三相共训
- [GEN-1.5 一次示范学习（Physical Prompting）](../entities/generalist-gen15-one-shot.md) — 人→机 / 仿真 prompt→真机（in-context 语义）
- [HOST](../entities/paper-host-one-shot-human-video.md) — 人视频→机器人未来观测再出动作；开源对照
- [GEN-1 千手：跨末端执行器泛化](../entities/generalist-gen1-thousand-hands.md) — 末端接口多样性（产业闭源对照）
- [Gemini Robotics](../entities/gemini-robotics.md) — On-Device 2 快速跨本体叙事（权重未开源）
- [X-Morph](../entities/paper-xmorph.md) — 人体运动作非人形腿式可复用行为先验
- [UCAG-P](../entities/paper-ucag-p.md) — 操作 VLA：相机可观测锚点作跨本体共享动作目标
- [CrossTracer](../entities/paper-crosstracer.md) — 导航：像素轨迹残差适配轮式/腿式（NaviTrace；待核实开源）
- [具身大模型与本体协同设计](../concepts/embodied-foundation-model-hardware-codesign.md) — 模型通用 vs 硬件形态通用

## 参考来源

- 本库归纳自 [Cross-Embodiment Transfer Strategy](../queries/cross-embodiment-transfer-strategy.md) 及 motion-retargeting / sim2real 交叉页
- [UCAG-P 论文摘录](../../sources/papers/ucag_p_arxiv_2608_26058.md) — 相机系锚点作跨本体共享动作目标
- [CrossTracer 论文摘录](../../sources/papers/crosstracer_arxiv_2608_06688.md) — 导航像素轨迹残差（与 WBT 换骨架正交）
- [GEN-1.5: Embodied Foundation Models are One-Shot Learners（来源归档）](../../sources/blogs/generalist_gen15_one_shot.md) — physical prompting / 人→机与 sim→真机提示
- [Towards Machines with a Thousand Hands（来源归档）](../../sources/blogs/generalist_thousand_hands.md) — 跨末端「千手」产业样本
- [Gemini Robotics 2 全身智能（来源归档）](../../sources/blogs/gemini_robotics_2_whole_body.md) — On-Device 跨本体适配声明
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`cross-embodiment` 命中规则）
