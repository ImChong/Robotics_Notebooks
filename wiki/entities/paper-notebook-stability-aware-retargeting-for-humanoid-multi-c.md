---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2510.04353"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_stability-aware-retargeting-for-humanoid-multi-c.md
summary: "当人形机器人需要用手去推墙、撑天花板、按在斜面/不平表面上完成作业时，多了几个接触点反而让稳定性更难算、更易崩——操作员一个看似合理的指令就可能让某个关节力矩饱和或让手打滑。本文提出稳定性感知的重定向（stability-aware retargeting）：用 actuation-aware 的质心稳定区域度量「现在离失稳还有多远」，再解析地算出稳定裕度对接触点/关节位形的梯度，从而在遥操作回路里实时、低开销地微调接触位置与上身姿态，把机器人往更稳的位形拉，同时仍尊重操作员的高层意图。"
---

# Stability-Aware Retargeting for Humanoid Multi-Contact Teleoperation

**Stability-Aware Retargeting for Humanoid Multi-Contact Teleoperation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

当人形机器人需要用手去推墙、撑天花板、按在斜面/不平表面上完成作业时，多了几个接触点反而让稳定性更难算、更易崩——操作员一个看似合理的指令就可能让某个关节力矩饱和或让手打滑。本文提出稳定性感知的重定向（stability-aware retargeting）：用 actuation-aware 的质心稳定区域度量「现在离失稳还有多远」，再解析地算出稳定裕度对接触点/关节位形的梯度，从而在遥操作回路里实时、低开销地微调接触位置与上身姿态，把机器人往更稳的位形拉，同时仍尊重操作员的高层意图。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| CoM | Center of Mass | 质心 |
| LP | Linear Program | 线性规划（求稳定区域边界用） |
| IK | Inverse Kinematics | 逆运动学 |
| RA-L | Robotics and Automation Letters | IEEE 机器人快报 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- **多接触作业落地**：给"用手撑环境做活"的遥操作提供了实时稳定性护栏，是 loco-manipulation 走向真实场景的关键一环
- **可微稳定性度量**：把"离失稳多远"做成可微标量并高效求梯度，可被 RL 奖励、MPC 代价、共享自治等多处复用
- **共享自治(shared autonomy)范式**：示范了"听人 + 自动兜底"的好分工：人给意图，系统在零空间里保平衡
- **与算法/界面路线互补**：同模块 [SEW-Mimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation/SEW-Mimic__Closed-Form_Geometric_Retargeting_Solver_for_Upper_Body_Humanoid_Teleoperation.html) 解决"重定向到哪"，本文解决"重定向时别失稳"

## 解决什么问题

遥操作人形机器人做 loco-manipulation 时，常需要**手部接触环境**来辅助平衡或施力（撑墙、按住、推门）。但一旦接触面**不共面、不规则**，问题就来了：

- **失稳与打滑**：手的接触力方向受限于摩擦锥，操作员随手给的指令可能让接触点滑掉； - **电机力矩饱和**：多接触下重力/外力如何在关节间分配很敏感，某个关节很容易先到力矩极限，机器人"使不上劲"； - **操作员看不见这些约束**：VR 里的人只知道"我想把手放那儿"，对机器人当前**离失稳/饱和有多近**毫无感知。

## 核心机制

1. **稳定裕度的解析梯度**：用 LP 灵敏度分析一次性算出稳定裕度对接触点/位形的方向梯度，避免反复重解 LP，做到 kHz 级实时。
2. **稳定性感知重定向框架**：把梯度直接嵌进遥操作 IK/QP，**接触点 + 上身姿态**双管齐下扩大稳定区域。
3. **不抢操作员控制权的耦合方式**：零空间投影 + 三档策略 + 迟滞，让"稳定性改善"作为操作员意图的补充而非覆盖。
4. **仿真 + 真机双验证**：在 IHMC 液压-电驱人形上完成多接触取物，量化证明裕度提升与抗扰/力矩裕度的相关性。

方法拆解（深读笔记小节）：稳定性度量：actuation-aware 质心稳定区域；核心招式：稳定裕度的解析梯度（LP 灵敏度分析）；重定向：把梯度注入遥操作 IK；何时改写操作员指令（三档策略 + 迟滞）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation.html> |
| arXiv | <https://arxiv.org/abs/2510.04353> |
| 机构 | **IHMC**（Florida Institute for Human and Machine Cognition，佛罗里达人机认知研究所） |
| 作者 | **Stephen McCrory**, Romeo Orsolino, Dhruv Thanki, Luigi Penco, **Robert Griffin** |
| 发表 | 2025-10-05（arXiv） |
| 源码 | 截至当前未见公开仓库（论文未给出 GitHub / 项目页链接） |
| 笔记阅读日期 | 2026-06-16 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_stability-aware-retargeting-for-humanoid-multi-c.md](../../sources/papers/humanoid_pnb_stability-aware-retargeting-for-humanoid-multi-c.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation.html>
- 论文：<https://arxiv.org/abs/2510.04353>

## 推荐继续阅读

- [机器人论文阅读笔记：Stability-Aware Retargeting for Humanoid Multi-Contact Teleoperation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation.html)
