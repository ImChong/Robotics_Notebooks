---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2601.09031"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_rgmp-s.md
summary: "RGMP-S 把\"人形机器人做长程操作\"拆成两段：上层让 VLM 在轻量级 2D 几何先验的帮助下\"看懂场景 → 选对技能 → 拆分任务\"；下层让一种递归自适应脉冲网络（RASNet）在稀疏示范下学到时间一致的动作，避免过拟合。ManiSkill2 + 3 个真机平台上验证有效。"
---

# Generalizable Geometric Prior and Recurrent Spiking Feature Learning for Humanoid Robot Manipulation

**Generalizable Geometric Prior and Recurrent Spiking Feature Learning for Humanoid Robot Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

RGMP-S 把"人形机器人做长程操作"拆成两段：上层让 VLM 在轻量级 2D 几何先验的帮助下"看懂场景 → 选对技能 → 拆分任务"；下层让一种递归自适应脉冲网络（RASNet）在稀疏示范下学到时间一致的动作，避免过拟合。ManiSkill2 + 3 个真机平台上验证有效。

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
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2601.09031> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**RGMP-S 的取舍是「上层借现成模型泛化、下层省示范数据」：把场景理解与任务拆分交给 VLM 加轻量 2D 几何先验，把稀疏示范下的时序一致性交给递归脉冲网络，而不是用一个端到端大模型硬吃长程操作。**

- 上层真正起作用的不只是 VLM，而是给它补的**轻量级 2D 几何先验**——它把「看懂场景」落到可选技能与可拆分的子任务上。
- 下层 RASNet 的目标很具体：在**稀疏示范**下学到时间一致的动作并抑制过拟合，这正是长程任务里最容易崩的一段。
- 验证覆盖 ManiSkill2 仿真与 3 个真机平台，说明这套分层配方不是只在单一平台成立；但具体成功率与消融本页未给出。
- 本页为策展索引级摘要，量化 benchmark 与实机指标以深读笔记和论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_rgmp-s.md](../../sources/papers/humanoid_pnb_rgmp-s.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2601.09031>

## 推荐继续阅读

- [机器人论文阅读笔记：Generalizable Geometric Prior and Recurrent Spiking Feature Learning for Humanoid Robot Manipulation](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation/RGMP-S__Generalizable_Geometric_Prior_and_Recurrent_Spiking_Feature_Learning_for_Humanoid_Manipulation.html)
