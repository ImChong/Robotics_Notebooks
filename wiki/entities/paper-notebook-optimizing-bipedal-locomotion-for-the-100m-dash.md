---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2508.03070"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_optimizing-bipedal-locomotion-for-the-100m-dash.md
summary: "Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running

**Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见上方「流程总览」（依据论文方法部分，非深读笔记）。

## 流程总览

```mermaid
flowchart TD
    A["Cassie 盲走控制器<br/>LSTM 策略 40Hz 输出 PD 目标<br/>PD 控制器 2kHz"] --> B["III-A 训练覆盖步态参数谱<br/>ratio 与 freq 在手调映射附近均匀采样<br/>速度 0 至 5 m/s，PPO + MuJoCo"]
    B --> C["III-B 仿真网格搜索<br/>各速度下遍历 ratio × freq"]
    C --> D["四项代价打分<br/>速度误差 / 运输成本 / 力矩 / 电机速度<br/>摔倒记 0 分"]
    D --> E["各速度取 top-5 参数组合<br/>得到速度到 ratio, freq 的映射"]
    E --> F["IV-A 百米冲刺特化<br/>只训练非负速度、仅保留前向速度指令"]
    F --> G["跑步策略 + 站立策略<br/>两个策略"]
    subgraph 阶段["百米五阶段切换"]
        S1["阶段1 站立<br/>站立策略"] -->|"IV-B 在 GRF 交叉相位<br/>设定固定相位并切换"| S2["阶段2-3 加速与稳速跑<br/>跑步策略"]
        S2 -->|"IV-C 等待脚到达最高点相位<br/>速度指令降为小负值后切换"| S3["阶段4 减速踏步<br/>跑步策略"]
        S3 --> S4["阶段5 站立<br/>站立策略"]
    end
    G --> S1
```

图依据：论文 §II、§III-A/B、§IV-A/B/C（Fig.3、Fig.5、Fig.6）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/optimizing-bipedal-locomotion-for-the-100m-dash` |
| arXiv | <https://arxiv.org/abs/2508.03070> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 05_Locomotion 下的待读条目：只登记了这篇「百米冲刺双足运动优化 + 与人类跑步对比」的存在，优化方法与对比结论均待深读笔记补齐。**

- 页面可确证的只有分类归属、PROGRESS.md 上的「待深读」状态与 arXiv 编号 2508.03070；速度指标、优化目标与人机对比数据全部无记录。
- 适用边界：可用于分类检索，不能作为速度成绩或人机差距的引用来源。
- 深读状态以上游 PROGRESS.md 为准；笔记完成后本页需补成完整摘要并深化归纳。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_optimizing-bipedal-locomotion-for-the-100m-dash.md](../../sources/papers/humanoid_pnb_optimizing-bipedal-locomotion-for-the-100m-dash.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2508.03070>
- [arXiv PDF](https://arxiv.org/pdf/2508.03070)：已通读方法部分（实验与附录以略读为准）：§II Background、§III Optimizing Gait Parameters、§IV Comparison / 百米冲刺策略切换（IV-A 至 IV-C）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
