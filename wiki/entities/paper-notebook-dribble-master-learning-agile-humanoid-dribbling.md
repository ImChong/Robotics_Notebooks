---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2505.12679"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dribble-master-learning-agile-humanoid-dribbling.md
summary: "Dribble Master：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Dribble Master

**Dribble Master: Learning Agile Humanoid Dribbling Through Legged Locomotion** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Dribble Master 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，本页只保留分类位置与原文入口；方法模块与数据流见下方「流程总览」（仅依据论文方法部分，不含实验结论）。

## 流程总览

```mermaid
flowchart TD
    subgraph SIM["仿真训练（Isaac Gym / Humanoid Gym，PPO 非对称 Actor-Critic）"]
        S1["阶段一：运动技能<br/>球初始远（约 10 m）；虚拟相机 FOV 放大为 2 倍；<br/>高权重运动 / 追球奖励，无球速跟踪奖励"]
        S2["阶段二：精细控球<br/>球初始近（2 m 内）；FOV 恢复 D455 真实值；<br/>高权重运球奖励，去除参考动作奖励"]
        S1 -->|"人工切换"| S2
    end
    VC["虚拟相机模型（RealSense D455 参数）：球出视野则停止更新球位置，保留 0.3 s 历史"] --> OBS
    SR["Sim-to-real 措施：电机延迟 0-20 ms、粗糙地形、球位置噪声、域随机化"] --> SIM
    OBS["观测：球速度指令 v_cmd、本体感知（q, q̇, 姿态）、球位置 b + 视野内指示、正弦时钟信号"] --> SIM
    SIM --> ACT["Actor MLP [512,256,128]（Critic 额外使用特权信息）"]
    ACT --> DEP["部署：Booster T1，策略 50 Hz"]
    CAM["RealSense D455 RGB 图像"] --> YOLO["YOLOv8 球检测（30 Hz，TensorRT）"]
    YOLO --> EXT["相机-机器人外参变换 → 基座系球位置"]
    EXT --> DEP
    JOY["操纵杆：全局系球速度指令 (vx, vy)"] --> DEP
    DEP --> OUT["14 维动作：2-DoF 头 + 12-DoF 腿的目标关节位置"]
    OUT --> PD["PD 控制器 → 力矩"]
```

图依据：论文 §III-A 至 §III-E、Fig.2 / Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/dribble-master-learning-agile-humanoid-dribbling` |
| arXiv | <https://arxiv.org/abs/2505.12679> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Dribble Master 在本库目前只是一条「已排期、未深读」的待读条目：它保证"人形敏捷球类技能"这一话题在 locomotion 分支下不缺节点，但还给不出可引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2505.12679、分类 05_Locomotion 与计划中的笔记文件夹；标题指向"通过足式运动学习敏捷人形运球"这一命题，奖励设计、球体交互建模与 sim2real 细节页内一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-05-locomotion.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为方案对比或复现依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 PROGRESS.md 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dribble-master-learning-agile-humanoid-dribbling.md](../../sources/papers/humanoid_pnb_dribble-master-learning-agile-humanoid-dribbling.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.12679>
- [arXiv PDF](https://arxiv.org/pdf/2505.12679)：已通读方法部分（实验与附录以略读为准）：§III Methods 全部（问题表述、两阶段课程、虚拟相机、仿真环境、真机部署）；§IV 仅略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
