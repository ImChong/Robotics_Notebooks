---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.14266"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_spi-active.md
summary: "高精度腿足技能（如精准落点的跳跃）对 sim-real gap 极其敏感——差一点动力学参数，跳跃就偏几十厘米。主流做法域随机化（DR）靠\"把未知量全随机化\"求鲁棒，但会让策略偏保守、且难以精确。传统系统辨识（SysID）又常假设动力学可微、能直接测扭矩，这些在富接触腿足系统里根本不成立。SPI-Active 给出两阶段方案：① SPI——用 GPU 上的大规模并行采样（CMA-ES）最小化\"仿真 vs 真实\"轨迹误差，反推质量-惯量与电机扭矩参数；② Active——不再被动采数据，而是优化探索策略的指令序列去最大化 Fisher 信息（等价 D-最优实验设计），专门激发\"最能暴露参数\"的高扭矩步态，再回炉重新辨识。最终高精度技能零样本迁移，较基线提升 42–63%。"
---

# Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning

**Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

高精度腿足技能（如精准落点的跳跃）对 sim-real gap 极其敏感——差一点动力学参数，跳跃就偏几十厘米。主流做法域随机化（DR）靠"把未知量全随机化"求鲁棒，但会让策略偏保守、且难以精确。传统系统辨识（SysID）又常假设动力学可微、能直接测扭矩，这些在富接触腿足系统里根本不成立。SPI-Active 给出两阶段方案：① SPI——用 GPU 上的大规模并行采样（CMA-ES）最小化"仿真 vs 真实"轨迹误差，反推质量-惯量与电机扭矩参数；② Active——不再被动采数据，而是优化探索策略的指令序列去最大化 Fisher 信息（等价 D-最优实验设计），专门激发"最能暴露参数"的高扭矩步态，再回炉重新辨识。最终高精度技能零样本迁移，较基线提升 42–63%。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| Sim2Real | Sim-to-Real | 仿真训练，真实部署 |
| DR | Domain Randomization | 域随机化：随机化物理参数以求鲁棒 |
| SysID | System Identification | 系统辨识：从数据反推物理参数 |
| SPI | Sampling-based Parameter Identification | 采样式参数辨识（本文第一阶段） |
| FIM | Fisher Information Matrix | Fisher 信息矩阵，刻画参数可辨识程度 |
| D-optimality | D-最优 | 一种最优实验设计准则：最小化参数估计协方差 |
| CMA-ES | Covariance Matrix Adaptation Evolution Strategy | 无梯度进化优化算法 |
| CoM | Center of Mass | 质心 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 解决什么问题

**问题陈述**：腿足机器人要做**高精度**任务（精准落点的前跳、偏航跳、精确速度/姿态跟踪），对动力学参数误差极其敏感——名义模型和真机差一点，误差就被放大到几十厘米。而弥合 sim-real gap 的两条主流路线都有硬伤：

1. **域随机化（DR）**：把摩擦、质量、增益、延迟大范围随机化。结果策略被迫**保守**，牺牲精度；且随机化范围靠启发式手调。 2. **传统 SysID**：多假设**动力学可微**、能**直接测量关节扭矩**。这些前提在**富接触**腿足系统里不成立——接触不可微、真机也没有力矩传感器。

## 核心机制

1. **两阶段辨识范式**：SPI（采样式辨识）+ Active（主动探索），把"辨识"与"最优数据采集"耦合成闭环，用**辨识**替代**盲目域随机化**；
2. **无需可微动力学/力矩传感器**：CMA-ES 采样式优化天然适配富接触、不可微系统，仅用标准状态轨迹即可辨识质量-惯量与电机扭矩参数；
3. **把最优实验设计引入腿足 SysID**：以 **D-最优（最小化 FIM 之逆迹）** 为目标，**优化指令序列**而非从零学探索策略，专门激发高信息量数据；
4. **高精度技能实证**：Unitree Go2（含 33% 体重负载）完成精准前跳/偏航跳/速度/姿态跟踪，较基线提升 **42–63%**，前跳落点误差低至 ~3.6 cm；并在 G1 人形上验证泛化；
5. **工程价值**：🌟 全套代码开源（[LeCAR-Lab/SPI-Active](https://github.com/LeCAR-Lab/SPI-Active)），可复用于新平台的辨识流水线。

方法拆解（深读笔记小节）：阶段 1 · SPI —— 采样式参数辨识；阶段 2 · Active —— 主动探索（最大化 Fisher 信息）；闭环。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html> |
| arXiv | <https://arxiv.org/abs/2505.14266> |
| 发表 | 2025-05-20 (arXiv) |
| 会议 | CoRL 2025 |
| 项目主页 | [lecar-lab.github.io/spi-active_](https://lecar-lab.github.io/spi-active_/) |
| 源码 | 🌟 [github.com/LeCAR-Lab/SPI-Active](https://github.com/LeCAR-Lab/SPI-Active) |
| 笔记阅读日期 | 2026-07-05 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_spi-active.md](../../sources/papers/humanoid_pnb_spi-active.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html>
- 论文：<https://arxiv.org/abs/2505.14266>

## 推荐继续阅读

- [机器人论文阅读笔记：Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html)
