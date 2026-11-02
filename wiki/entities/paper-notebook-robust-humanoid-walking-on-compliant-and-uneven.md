---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2504.13619"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robust-humanoid-walking-on-compliant-and-uneven.md
summary: "Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL

**Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），结论与指标以论文为准。

## 流程总览

```mermaid
flowchart TD
    subgraph OBS["策略输入 / 输出 §III"]
        I1["机器人状态: 根姿态 roll/pitch、角速度、关节位置/速度、电机电流"]
        I2["外部状态: 模式 (站立 / 原地踏步 / 前进) 独热 + 参考速度 + 周期时钟 clock"]
        A1["输出: 12 维腿部关节目标位置 → 低增益 PD (策略 40 Hz，PD 1000 Hz)<br/>上身关节固定"]
    end
    subgraph P1["训练阶段 1 §IV-C"]
        T1["平坦刚性地面训练基础策略<br/>站立 / 原地踏步 / 前进模式随机切换"]
    end
    subgraph P2["训练阶段 2 §IV"]
        T2["从预训练权重继续微调<br/>随机化脚部柔顺 solref；随机移动 MuJoCo 高度场地形 (双支撑期禁用)<br/>动力学参数 (阻尼、摩擦、质量、质心) 约每 0.5 s 重采样"]
    end
    I1 --> T1
    I2 --> T1
    T1 --> T2
    T2 --> PPO["PPO 训练 MLP 策略/价值网络 (2 层 x 256)"]
    PPO --> DEP["零样本迁移 HRP-5P 真机<br/>C++ mc-rtc 控制器，mc-mujoco 先行验证"]
    subgraph V["增强策略 §V 非周期步态"]
        V1["动作增至 13 维: 额外预测相位偏移 a_δϕ<br/>ϕ_t+1 = ϕ_t + clip(a_δϕ, -5, 5) + 1"]
    end
    T2 -.->|"策略输出增加 clock 调制 (Fig.2 右)"| V
```

图依据：论文 §III–§V / Fig.2（左：两阶段训练；右：预测 clock 调制的增强策略）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/robust-humanoid-walking-on-compliant-and-uneven` |
| arXiv | <https://arxiv.org/abs/2504.13619> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是该论文的待读条目：标题锁定的问题——柔顺且不平地形上的鲁棒行走——本库已认可其归属，但方法与证据尚未消化。**

- 已确定的信息仅为分类 05_Locomotion 与 arXiv 2504.13619；「实验与评测」明确留空。
- 柔顺地形（compliant）与不平地形（uneven）本是两类不同扰动，标题把它们并列，但本页没有依据说明论文如何区分处理，引用时不要替它下结论。
- 当前用途是待读条目与路由：上溯 [05_Locomotion 分类页](../overview/paper-notebook-category-05-locomotion.md) 与 [总索引](../overview/humanoid-paper-notebooks-index.md)，指标一律以 arXiv 原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robust-humanoid-walking-on-compliant-and-uneven.md](../../sources/papers/humanoid_pnb_robust-humanoid-walking-on-compliant-and-uneven.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2504.13619>
- [arXiv PDF](https://arxiv.org/pdf/2504.13619)：已通读方法部分（实验与附录以略读为准）：§III–§V 环境、训练与非周期步态（Fig.2）；§VI 实验结果略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
