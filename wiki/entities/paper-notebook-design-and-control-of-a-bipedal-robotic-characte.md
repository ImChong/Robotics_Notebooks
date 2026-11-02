---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2501.05204"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_design-and-control-of-a-bipedal-robotic-characte.md
summary: "Design and Control of a Bipedal Robotic Character：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Design and Control of a Bipedal Robotic Character

**Design and Control of a Bipedal Robotic Character** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Design and Control of a Bipedal Robotic Character 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记尚未撰写；方法流程见下方「流程总览」（依据论文方法部分），具体数字以论文为准。

## 流程总览

```mermaid
flowchart TD
    subgraph 离线["离线：设计与训练"]
        A["机械设计 ⇄ 动画工具迭代<br/>(程序化步态生成工具)"] --> B["机械设计 → 仿真模型<br/>+ 执行器模型 + 域随机化 (Isaac Gym)"]
        A --> C["动画工具导出运动学参考<br/>(持续 perpetual / 周期 periodic / 分段 episodic)"]
        C --> D["参考生成器 f 与路径帧<br/>+ 命令 g_t 定义"]
        B --> E["PPO 训练，每种运动一个策略<br/>奖励=模仿+正则+存活，g_t 全范围随机"]
        D --> E
        E --> F["策略：站立(perpetual) / 行走(periodic)<br/>/ 多个 episodic 情绪动作"]
    end
    subgraph 运行["运行时"]
        G["操作员手持遥控<br/>(摇杆 / 按钮 / 策略切换)"] --> H["动画引擎<br/>背景动画 + 触发动画 + 摇杆层"]
        H -->|"控制命令 g_t / 策略切换"| I["冻结策略 50 Hz<br/>输出关节位置设定点"]
        H -->|"show function 信号"| J["天线 / LED 眼 / 头灯"]
        H -->|"同步播放命令"| K["音频引擎"]
        I --> L["一阶保持 + 低通滤波<br/>PD 控制 600 Hz 驱动执行器"]
        L --> M["IMU + 执行器测量<br/>状态估计 s_t"]
        M --> I
    end
    F --> I
```

图依据：论文 §III 总览 / Fig.2、§V 强化学习、§VI 运行时（Fig.5 动画管线）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/design-and-control-of-a-bipedal-robotic-characte` |
| arXiv | <https://arxiv.org/abs/2501.05204> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Design and Control of a Bipedal Robotic Character 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2501.05204、分类 12_Hardware_Design、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-12-hardware-design.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_design-and-control-of-a-bipedal-robotic-characte.md](../../sources/papers/humanoid_pnb_design-and-control-of-a-bipedal-robotic-characte.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2501.05204>
- [arXiv PDF](https://arxiv.org/pdf/2501.05204)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 总览、§IV 机电设计、§V 强化学习（A 动画输入、B 奖励、C 策略、D 底层控制、E 仿真）、§VI 运行时

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
