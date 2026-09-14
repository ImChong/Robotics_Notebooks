---
type: entity
tags: [paper, humanoid, granular-terrain, teacher-student, georgia-tech]
status: complete
updated: 2026-09-14
arxiv: "2609.10286"
related:
  - ../tasks/humanoid-locomotion.md
  - ../tasks/locomotion.md
  - ./paper-wm-loco.md
sources:
  - ../../sources/papers/granular_terrain_humanoid_arxiv_2609_10286.md
summary: "Granular Terrain Humanoid（arXiv:2609.10286）：3D resistance theory granular contact solver; Teacher-Student RL with VAE privileged terrain; zero-shot on basalt/dry sa；截至入库日未见官方代码。"
---

# Granular Terrain Humanoid（arXiv:2609.10286）

**Granular Terrain Humanoid**（*Learning Terrain-Adaptive Humanoid Locomotion on Granular Terrain*，[arXiv:2609.10286](https://arxiv.org/abs/2609.10286)）由 **佐治亚理工（Georgia Tech）；东北大学（Northeastern University）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

面向颗粒地形的人形机器人地形自适应运动学习 — 3D resistance theory granular contact solver。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VAE | Variational Autoencoder | 变分自编码器，压缩特权地形 |
| RL | Reinforcement Learning | 强化学习 |
| TS | Teacher-Student | 教师-学生蒸馏框架 |

## 为什么重要

松散颗粒地形接触力学与刚地差异大；显式颗粒模型 + 特权信息蒸馏是 sim2real 可行路径。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 佐治亚理工（Georgia Tech）；东北大学（Northeastern University） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

用 3D resistance theory 构建颗粒接触求解器供仿真 Teacher；Student 策略在 VAE 潜空间吸收特权地形特征，部署时仅依赖本体与有限感知。

### 流程总览

```mermaid
flowchart LR
  sim[颗粒仿真+特权地形] --> teacher[Teacher RL]
  teacher --> vae[VAE 地形编码]
  vae --> student[Student 策略]
  student --> real[玄武岩/干沙/海滩沙]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 复现时先验证颗粒求解器与刚地 baseline 差距；Student 输入维与 VAE 瓶颈需与真机传感对齐。 |

## 实验与评测

多类真实颗粒地形零样本行走；对比无颗粒模型与无 VAE 的 ablation（见论文）。

## 结论

颗粒接触建模与特权地形蒸馏使人形在多种松散地面上零样本可行。

1. 3D 阻力颗粒求解器是仿真可信度的关键。
2. VAE 承载特权地形而非直接喂 Student 全状态。
3. 玄武岩/干沙/海滩沙零样本验证 sim2real。
4. Teacher-Student 降低 onboard 传感负担。
5. 颗粒参数辨识误差会放大到真机滑移。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 刚地 RL 行走 | 无颗粒力学，松散面易陷/滑 |
| 盲走策略 | 无地形自适应，颗粒面失败率高 |

## 局限与风险

颗粒物性域随机范围未完全覆盖所有户外沙型；高湿海滩沙与仿真差距待量化。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [locomotion](../tasks/locomotion.md)
- [./paper-wm-loco.md](./paper-wm-loco.md)

## 参考来源

- [granular_terrain_humanoid_arxiv_2609_10286.md](../../sources/papers/granular_terrain_humanoid_arxiv_2609_10286.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.10286](https://arxiv.org/abs/2609.10286)
