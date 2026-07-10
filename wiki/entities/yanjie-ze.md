---
type: entity
tags: [humanoid, teleoperation, loco-manipulation, motion-retargeting, imitation-learning, visual-rl, stanford, sjtu, amazon, figure-ai]
status: complete
updated: 2026-06-25
related:
  - ./paper-twist.md
  - ./paper-twist2.md
  - ./paper-resmimic.md
  - ./paper-notebook-visualmimic.md
  - ../methods/motion-retargeting-gmr.md
  - ../concepts/motion-retargeting.md
  - ../tasks/teleoperation.md
  - ../tasks/loco-manipulation.md
  - ./xue-bin-peng.md
  - ./tairan-he.md
sources:
  - ../../sources/sites/yanjieze.md
summary: "迮炎杰（Yanjie Ze）为 Stanford CS 博士生，研究人形通向具身通用智能的完整路径；代表作含 GMR 重定向、TWIST/TWIST2 全身遥操作与数据采集、VisualMimic/ResMimic 视觉 loco-manipulation，主页为论文与开源项目总索引。"
---

# Yanjie Ze（迮炎杰）

## 一句话定义

**Yanjie Ze** 是面向 **人形机器人模仿学习与 loco-manipulation** 的研究者：从 SJTU 阶段的 **3D 视觉 RL / 扩散策略**，到 Stanford 与 Amazon FAR 合作网络中的 **GMR 重定向 → TWIST 全身跟踪 → TWIST2 可扩展数据采集 → VisualMimic / ResMimic 视觉全身操作**，形成社区可见的 **「人体运动 → 机器人轨迹 → 策略学习」闭环技术线**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GMR | General Motion Retargeting | 实时通用人体→人形运动重定向前端 |
| WBT | Whole-Body Tracking | 全身运动跟踪 RL 低层控制 |
| MoCap | Motion Capture | 遥操作与演示数据的常见来源 |
| VLA | Vision-Language-Action | 视觉–语言–动作多模态策略范式 |
| FAR | Frontier AI & Robotics | Amazon 机器人研究团队（TWIST2 等合作方） |

## 为什么重要

- **与仓库人形主线高度重合**：其公开工作串覆盖 [运动重定向](../concepts/motion-retargeting.md)、[遥操作](../tasks/teleoperation.md)、[Loco-Manipulation](../tasks/loco-manipulation.md) 与 [GMR](../methods/motion-retargeting-gmr.md)，是理解 **2024–2026 人形模仿学习论文潮** 的 **作者级索引**。
- **工程可检索**：个人页与 GitHub 集中给出项目站、开源代码与 BibTeX，比零散抓 arXiv 更适合作为 ingest 溯源与后续 curator 补链入口。
- **跨机构协作节点**：SJTU（Xiaolong Wang、Huazhe Xu）→ Stanford（Jiajun Wu、C. Karen Liu）→ Amazon FAR（Guanya Shi、Pieter Abbeel、Rocky Duan）→ Figure AI Helix AI，便于与 [Tairan He](./tairan-he.md)、[Xue Bin Peng](./xue-bin-peng.md) 等同期人形研究者互链。

## 核心研究脉络（归纳）

1. **重定向前端（GMR）**：把异构人体运动（MoCap、视频姿态、SMPL 等）实时映射到人形关节轨迹，是 TWIST / VisualMimic 等管线的 **共享基础设施**。
2. **全身遥操作与跟踪（TWIST）**：以 RL 运动跟踪为低层，把 **遥操作演示** 变成可规模化训练的数据生产方式（CoRL 2025；CVPR Humanoid Agents Workshop Best Demo）。
3. **可扩展数据采集（TWIST2）**：便携全身遥操作 + 颈增广 egocentric 感知 + 分层 visuomotor（System1 跟踪 + System2 模仿），面向 **长时程 loco-manipulation 数据集**（ICRA 2026 Oral）。
4. **视觉全身技能（VisualMimic、ResMimic）**：在跟踪先验之上推进 **RGB 驱动的 loco-manipulation** 与 **残差学习整身操作**。

## 流程总览（反复出现的数据–控制闭环）

下列图只描述其多篇论文共享的 **逻辑骨架**，模块细节以各论文为准。

```mermaid
flowchart LR
  Human["人体运动\nMoCap / 视频 / 遥操作"] --> GMR["GMR 重定向\n实时关节轨迹"]
  GMR --> Track["低层 WBT\nRL 运动跟踪"]
  Track --> Data["TWIST2 数据采集\negocentric + 全身流"]
  Data --> Policy["高层策略\n扩散 / 残差 / 视觉模仿"]
  Policy --> Deploy["真机 loco-manipulation"]
```

## 常见误区或局限

- **主页 ≠ 最新事实源**：论文接收状态、实习任职以 **论文页与机构新闻** 为准；本页不替代一手引用。
- **GMR 缩写歧义**：本知识库 **GMR** 均指 *General Motion Retargeting*，与统计学高斯混合回归无关（见 [GMR 方法页](../methods/motion-retargeting-gmr.md)）。
- **合著分工**：VisualMimic、Retargeting Matters 等存在共同一作，阅读时应区分 **个人贡献边界**。

## 关联页面

- [TWIST（全身遥操作模仿）](./paper-twist.md)
- [TWIST2（可扩展全身数据采集）](./paper-twist2.md)
- [ResMimic](./paper-resmimic.md)
- [VisualMimic](./paper-notebook-visualmimic.md)
- [GMR：通用动作重定向](../methods/motion-retargeting-gmr.md)
- [运动重定向](../concepts/motion-retargeting.md)
- [遥操作](../tasks/teleoperation.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Tairan He（何泰然）](./tairan-he.md)
- [Xue Bin Peng（彭学斌）](./xue-bin-peng.md)

## 参考来源

- [Yanjie Ze 个人主页原始资料](../../sources/sites/yanjieze.md)

## 推荐继续阅读

- [机器人论文阅读笔记：TWIST](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TWIST__Teleoperated_Whole-Body_Imitation_System/TWIST__Teleoperated_Whole-Body_Imitation_System.html)
- [TWIST2 项目页](https://yanjieze.com/projects/TWIST2/)
- [GMR 官方仓库](https://github.com/YanjieZe/GMR)
- [Awesome Humanoid Robot Learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning)
