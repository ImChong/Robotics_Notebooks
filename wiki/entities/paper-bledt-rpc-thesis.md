---
type: entity
tags: [paper, thesis, mpc, locomotion, quadruped, mit, control]
status: complete
updated: 2026-07-25
related:
  - ./paper-extracting-legged-locomotion-heuristics-rpc.md
  - ./paper-robust-autonomous-navigation-mini-cheetah-vision.md
  - ./paper-wbic-mpc-mini-cheetah.md
  - ./mit-mini-cheetah.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/bledt_rpc_thesis_mit_2020.md
  - ../../sources/blogs/robot_daycare_mini_cheetah_2019.md
summary: "Bledt MIT 博士论文（2020）：Regularized Predictive Control（RPC）框架——用正则启发式增强动态腿足鲁棒预测控制。"
---

# Regularized Predictive Control Framework（Bledt Thesis）

## 一句话定义

**Gerardo Bledt（MIT，2020 博士论文，[dspace:1721.1/125485](https://dspace.mit.edu/handle/1721.1/125485)）** 系统提出 **Regularized Predictive Control（RPC）**：在预测控制中引入可设计、可适应的**正则启发式**，提升动态腿足在模型简化与参数不确定下的鲁棒性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RPC | Regularized Predictive Control | 本文框架名 |
| MPC | Model Predictive Control | RPC 的预测控制母体 |
| WBIC | Whole-Body Impulse Control | 常与 RPC 组成导航系统下层 |
| SRBD | Single Rigid Body Dynamics | 简化模型来源之一 |
| TO | Trajectory Optimization | 相关离线探索工具 |

## 为什么重要

- 给 Mini Cheetah「模型基控制调参痛苦」一个方法论出口：启发式可提取、可在线适应。
- 直接支撑 [ICRA 2020 启发式提取](./paper-extracting-legged-locomotion-heuristics-rpc.md) 与 [IROS 2020 Vision 导航](./paper-robust-autonomous-navigation-mini-cheetah-vision.md)。
- 在博文论文清单中作为 RPC 线的总纲。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 麻省理工（MIT） |
| **类型** | PhD dissertation |
| **开源** | PDF 公开；实现能力见 Cheetah-Software / 后续论文实验 |

## 核心原理

- 预测控制性能强依赖代价与启发式；手工调参难且不可迁移。
- RPC：把正则项/启发式当作一等公民——离线探索 → 提取简单模型 → 在线适应。
- 目标：在不牺牲物理直觉的前提下逼近复杂动力学、吸收模型误差。

## 源码运行时序图

**不适用**（学位论文本身不是可运行软件发布；控制实现见 Cheetah-Software 与后续 RPC 论文）。

## 工程实践

| 项 | 建议 |
|----|------|
| 阅读顺序 | 本论著章节结构 → ICRA 2020 启发式短文 → Vision 导航系统文 |
| 实现 | 在现有 MPC 代价上增加可辨识正则，而非推倒重来 |
| 验证 | 对比「仅调增益」与「启发式正则」在扰动/速度包络上的差异 |

## 评测

| 维度 | 要点 |
|------|------|
| 平台语境 | Mini Cheetah 动态 locomotion |
| 后续 | ICRA/IROS 论文给出硬件证据链 |

## 结论

**总判：** Bledt 论文是 Mini Cheetah **RPC 控制哲学**的长文定义；短会论文是可执行摘要。

- 真影响：正则启发式可系统提取与适应。
- 次要代价：离线探索成本；启发式仍可能过拟合仿真。
- 部署：与 WBIC 叠用做导航/高动态，见系统论文。

## 局限与风险

- 学位论文篇幅长，工程上优先读 ICRA 短文落地。
- 无单一官方「RPC 包」仓库名。

## 关联页面

- [Extracting heuristics](./paper-extracting-legged-locomotion-heuristics-rpc.md)
- [Robust navigation](./paper-robust-autonomous-navigation-mini-cheetah-vision.md)
- [WBIC+MPC](./paper-wbic-mpc-mini-cheetah.md)
- [MIT Mini Cheetah](./mit-mini-cheetah.md)

## 参考来源

- [论文归档](../../sources/papers/bledt_rpc_thesis_mit_2020.md)
- [博文清单](../../sources/blogs/robot_daycare_mini_cheetah_2019.md)

## 推荐继续阅读

- DSpace：<https://dspace.mit.edu/handle/1721.1/125485>
