---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned, mit]
status: planned
updated: 2026-07-25
venue: curated
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-low-cost-modular-actuator-katz.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/papers/humanoid_pnb_proprioceptive-actuator-design-in-the-mit-cheeta.md
  - ../../sources/papers/low_cost_modular_actuator_katz_mit_2018.md
summary: "Proprioceptive actuator design in the MIT Cheetah：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。低成本模块化落地见 Katz 2018 thesis。"
---

# Proprioceptive actuator design in the MIT Cheetah

**Proprioceptive actuator design in the MIT Cheetah: Impact mitigation and high‑bandwidth physical interaction for dynamic legged robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

Proprioceptive actuator design in the MIT Cheetah 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks **progress 待深读** 清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 在深读笔记完成前，本页作为 **占位子节点**，避免知识图谱缺失该论文实体。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/proprioceptive-actuator-design-in-the-mit-cheeta` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 MIT Cheetah 本体感受执行器一文的待深读占位节点；目前真正落地的信息只有它的定位——硬件设计线上的源头文献，以及它与 Katz thesis 的「设计理念 → 低成本落地」配对关系。**

- 归类为 12_Hardware_Design，标题点明的两个诉求是冲击缓解与高带宽物理交互，这决定了它在本库属于硬件本体线而非策略学习线。
- 与 [Katz Mini Cheetah 执行器 thesis](./paper-low-cost-modular-actuator-katz.md) 成对阅读：后者给出 COTS 电机 + 6:1 行星的低成本模块化落地路径。
- 深读笔记尚未撰写、「实验与评测」留空，且核心信息表未列 arXiv 行；本页不能作为任何扭矩密度、带宽或冲击指标的引用出处，检索需走 PROGRESS.md 与来源归档。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 低成本模块化落地（COTS 电机 + 6:1 行星）：[Katz Mini Cheetah 执行器 thesis](./paper-low-cost-modular-actuator-katz.md)

## 参考来源

- [humanoid_pnb_proprioceptive-actuator-design-in-the-mit-cheeta.md](../../sources/papers/humanoid_pnb_proprioceptive-actuator-design-in-the-mit-cheeta.md)
- [Katz 2018 thesis 归档](../../sources/papers/low_cost_modular_actuator_katz_mit_2018.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- [Katz · A Low Cost Modular Actuator（Mini Cheetah）](./paper-low-cost-modular-actuator-katz.md)
