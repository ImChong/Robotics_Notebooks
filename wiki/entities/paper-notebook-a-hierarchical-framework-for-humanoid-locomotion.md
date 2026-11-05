---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2512.00077"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_a-hierarchical-framework-for-humanoid-locomotion.md
summary: "A Hierarchical Framework for Humanoid Locomotion with Supernumerary Limbs：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# A Hierarchical Framework for Humanoid Locomotion with Supernumerary Limbs

**A Hierarchical Framework for Humanoid Locomotion with Supernumerary Limbs** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

A Hierarchical Framework for Humanoid Locomotion with Supernumerary Limbs 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里保留分类位置与原文入口，方法流程见下方「流程总览」，其余结论不给出。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["低层训练：DRL 步态（DeepMimic 式模仿学习）"]
        ref["LAFAN1 行走参考轨迹"] --> rew["模仿奖励 R_t<br/>survival + stability + imitation"]
        sim["MuJoCo/JAX 仿真<br/>Unitree H1 + 2×Kinova Gen3 SL"] --> rew
        rew --> ppo["PPO 更新 Actor-Critic π_θ"]
        ppo --> pol["低层步态策略 π_θ"]
        cur["课程调度器<br/>负载质量 m 递增 + 手臂姿态分 4 阶段"] -.调整任务难度.-> sim
    end
    subgraph DEPLOY["部署：分层控制"]
        pol --> legs["H1 腿部/躯干动作 a_t"]
        st["机器人状态 s_t"] --> est["状态估计：CoM 与 CoS<br/>单支撑取支撑脚中心 / 双支撑取 CoP 力加权"]
        est --> d["稳定指标 d_xy = CoM_xy − CoS_xy"]
        d --> tgt["目标臂姿 q_target = q_base − Kp·d_xy"]
        tgt --> pd["PD 控制器 → SL 臂力矩 τ_arm"]
        legs --> fuse["控制融合：腿用 π_θ，SL 臂由平衡器覆盖"]
        pd --> fuse
        fuse --> env["仿真环境"]
    end
    pol -.训练好的策略.-> DEPLOY
```

图依据：论文 §2.1.2 / §2.2 / §2.3 / Fig.2.2-2.4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/a-hierarchical-framework-for-humanoid-locomotion` |
| arXiv | <https://arxiv.org/abs/2512.00077> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前提供的是「待读条目」而不是论文解读：它把这条 05_Locomotion 条目挂进知识库以保持检索连通，实质技术判断要等深读笔记落地。**

- 页面已确认的信息仅有分类（05_Locomotion）、arXiv 编号与计划文件夹路径；分层框架的机制、量化 benchmark 与实机指标均尚未记录。
- 适用边界很窄：可用于按分类检索，不能作为该论文技术结论的引用依据。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_a-hierarchical-framework-for-humanoid-locomotion.md](../../sources/papers/humanoid_pnb_a-hierarchical-framework-for-humanoid-locomotion.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.00077>
- [arXiv PDF](https://arxiv.org/pdf/2512.00077)：已通读方法部分（实验与附录以略读为准）：§2.1-§2.4

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
