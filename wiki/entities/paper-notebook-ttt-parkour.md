---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, unitree]
status: stub
updated: 2026-07-16
arxiv: "2602.02331"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_ttt-parkour.md
summary: "TTT-Parkour 把\"对一段陌生地形拍 RGB-D 视频 → 前馈式快速重建出可仿真的网格 → 在仿真里对预训练好的跑酷策略做 ≤10 分钟的微调 → 直接零样本部署回真机\"做成了端到端流水线，让 Unitree G1 能在楔块、桩柱、箱子、梯形台、窄梁等极端地形上稳定通行，摆脱了\"只能在程序化生成的简单地形上训练\"这一根本限制。"
---

# TTT-Parkour

**TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

TTT-Parkour 把"对一段陌生地形拍 RGB-D 视频 → 前馈式快速重建出可仿真的网格 → 在仿真里对预训练好的跑酷策略做 ≤10 分钟的微调 → 直接零样本部署回真机"做成了端到端流水线，让 Unitree G1 能在楔块、桩柱、箱子、梯形台、窄梁等极端地形上稳定通行，摆脱了"只能在程序化生成的简单地形上训练"这一根本限制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour.html> |
| arXiv | <https://arxiv.org/abs/2602.02331> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**TTT-Parkour 把测试时训练变成部署前的一道工序：不再指望一个策略泛化到所有地形，而是对眼前这段真实地形花 ≤10 分钟重建并微调，再零样本部署回真机。**

- 真正起作用的是 **重建-微调闭环**：RGB-D 视频 → 前馈式快速网格重建 → 仿真内微调预训练跑酷策略 → 零样本回真机；前馈重建的速度是这条链路能压进 10 分钟的前提。
- 它解除的是一个具体限制——**只能在程序化生成的简单地形上训练**；因此楔块、桩柱、箱子、梯形台、窄梁这类极端地形才进入可通行范围（Unitree G1 实机验证）。
- 适用边界：属于 **部署前的地形适配**，需要先对目标地形取景重建，并不是行进途中的在线适应；对无法预先扫描或动态变化的地形不适用。
- 本页为 **索引级实体**，成功率、重建质量与微调时长的量化结果以深读笔记与论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_ttt-parkour.md](../../sources/papers/humanoid_pnb_ttt-parkour.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour.html>
- 论文：<https://arxiv.org/abs/2602.02331>

## 推荐继续阅读

- [机器人论文阅读笔记：TTT-Parkour](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour/TTT-Parkour__Rapid_Test-Time_Training_for_Perceptive_Robot_Parkour.html)
