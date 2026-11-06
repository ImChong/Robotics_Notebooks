---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2506.01563"
venue: "2025.06"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-pamor.md
sources:
  - ../../sources/papers/humanoid_pnb_hierarchical-intention-aware-expressive-motion-g.md
summary: "Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots

**Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    IN["多模态输入 X_I<br/>视觉 V_in + 语言 L_in<br/>+ 输入输出历史 H_I/O"] --> PI
    PRE["L_pre：预提示 + Few-shot 示例<br/>角色与任务 / CoT 输出格式 / V-A 象限映射表 / 安全规则"] --> PI
    subgraph HI["高层：意图感知模块 π_i（Qwen2.5-VL-32B，ICL + CoT）"]
        PI["VLM 推理"] --> OUT["结构化输出 O<br/>场景描述 d / 意图 i / 置信度 c<br/>Valence-Arousal / 动作原语 ⟨m⟩"]
        OUT --> SAFE["低置信度时回退为中性手势<br/>禁止激进或不安全动作"]
    end
    SAFE --> DART
    subgraph MID["中层：运动规划器 π_p"]
        DART["DART 文本到运动扩散模型<br/>以 ⟨m⟩ 与前一段运动为条件自回归生成<br/>12.5 FPS 滑动窗口，预测 8 帧"] --> RT["神经重定向网络 MLP<br/>SMPL 135 维 → Unitree G1 29 关节"]
    end
    RT --> WBC
    subgraph LOW["低层：RL 全身控制器 π_w"]
        WBC["观测：根状态、关节位置/速度、上一动作、参考 y_t<br/>输出 29 维目标关节角"] --> PD["PD 控制器驱动电机"]
    end
    PD --> ROBOT["Unitree G1 执行表达性手势"]
    ROBOT -. 状态反馈 s_t .-> WBC
```

图依据：论文 §III-A–D / Fig. 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Hierarchical_Intention-Aware_Expressive_Motion_Generation_for_Humanoid_Robots` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是待读条目而非论文归纳：它只登记了这条「意图感知的表达性动作生成」条目这一条，实质结论要等读完笔记。**

- 页面现状：status 为 planned，正文只有分类归属与 progress.json 待深读状态，没有可供提炼的机制、指标或实验内容。
- 从标题能读出的定位仅限于「分层 + 意图感知 + 人形表达性动作生成」，归入 04_Loco-Manipulation_and_WBC；分层如何切分、意图从哪来在本页均无依据。
- 使用方式：把本页当检索入口——分类页 [paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) 与 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)；要读实质内容请直接走原文。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hierarchical-intention-aware-expressive-motion-g.md](../../sources/papers/humanoid_pnb_hierarchical-intention-aware-expressive-motion-g.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2506.01563)：已通读方法部分（实验与附录以略读为准）：§I 引言、§II 相关工作、§III HIAER 框架（§III-A 概览、§III-B 意图感知模块、§III-C 表达性运动生成、§III-D 全身控制器）；§IV 实验未读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
