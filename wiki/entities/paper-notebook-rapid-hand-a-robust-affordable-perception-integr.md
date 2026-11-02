---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2506.07490"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_rapid-hand-a-robust-affordable-perception-integr.md
summary: "RAPID Hand：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# RAPID Hand

**RAPID Hand: A Robust, Affordable, Perception-Integrated, Dexterous Manipulation Platform for Generalist Robot Autonomy** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

RAPID Hand 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论；论文方法的模块流程见下方「流程总览」。

## 流程总览

```mermaid
flowchart TD
    subgraph HW["硬件：20-DoF 手 + 感知集成"]
        A["20 个舵机电机<br/>每指 4 DoF"] --> B["通用多指节驱动方案<br/>远端关节直驱 / 近端并联 MCP·CMC"]
        S1["腕部 RGBD 相机"] --> E["自研感知集成电子<br/>主控 + 同步模块"]
        S2["指尖压阻触觉<br/>96 taxel/指"] --> E
        S3["20 关节本体感知"] --> E
    end
    E --> H["硬件同步<br/>I2C 读触觉 + PWM 触发相机曝光<br/>多模态对齐 ≤7 ms"]
    H --> SA["空间对齐<br/>正运动学得到局部触觉点云<br/>相机标定对齐到视觉坐标系"]
    subgraph TO["高自由度遥操作接口"]
        T1["Apple Vision Pro<br/>捕获人手位姿"] --> T2["重定向优化<br/>共形对齐约束 + 接触感知耦合约束 + 时间平滑"]
        T2 --> T3["UR10e 机械臂 + RAPID Hand"]
    end
    T3 --> D["示教数据<br/>同步 RGB / 触觉 / 关节角 / taxel 位置"]
    SA --> D
    D --> P["全手视触觉扩散策略<br/>腕部图像 + 触觉 + 本体感知融合"]
    P --> O["预测未来 26-DoF 手臂+手关节轨迹<br/>10 Hz 部署"]
```

图依据：论文 §3.2 / §4.1 / §4.2 / Fig. 4–5

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/rapid-hand-a-robust-affordable-perception-integr` |
| arXiv | <https://arxiv.org/abs/2506.07490> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 RAPID Hand 的待读条目：一个面向通用机器人自主性的「鲁棒 + 可负担 + 感知一体化」灵巧操作平台，但本库尚未深读其硬件取舍与实测表现。**

- 可确认的只有定位：12_Hardware_Design 之下的灵巧手／操作平台工作，标题自陈的三个卖点是鲁棒性、成本可负担与感知集成。
- 这类硬件平台真正的看点是取舍——成本 vs 自由度 vs 传感覆盖——而本页尚未记录任何取舍细节与实测数据。
- 当前价值只是可检索；引用硬件参数或成本结论前请回到论文原文。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_rapid-hand-a-robust-affordable-perception-integr.md](../../sources/papers/humanoid_pnb_rapid-hand-a-robust-affordable-perception-integr.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2506.07490>
- [arXiv PDF](https://arxiv.org/pdf/2506.07490)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 平台（§3.1 手部本体、§3.2 感知框架）、§4 学习灵巧技能（§4.1 遥操作、§4.2 视触觉策略）；§5 实验仅略读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
