---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2506.00043"
venue: "2025.06"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_from-motion-to-behavior.md
summary: "From Motion to Behavior：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# From Motion to Behavior

**From Motion to Behavior: Hierarchical Modeling of Humanoid Generative Behavior Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

From Motion to Behavior 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    TXT["文本提示<br/>如：指挥家在舞台上指挥乐队"] --> LLM
    subgraph TP["任务规划（Task Planning）"]
        LLM["LLM 行为规划器<br/>朴素 CoT，补充动作幅度与速度等运动学属性"] --> SCR["组合脚本 Composed Script<br/>PoseScript p_i（关键帧）与<br/>MotionScript a_i（帧间过渡）交替"]
        SCR --> VAE["文本到姿态 VAE（取自 PoseScript）<br/>文本编码器 E → 潜变量 z_i → 姿态解码器 D<br/>得到 SMPL 姿态 x_i"]
    end
    VAE --> DIF
    subgraph MP["运动规划（Motion Planning）"]
        DIF["条件扩散模型 φ_θ（运动插帧）<br/>以 SMPL 姿态序列与 MotionScript 为条件<br/>生成离散参考轨迹 M"] --> CTRL["物理感知控制器<br/>如 PHC / HOVER，MPC 或 RL 策略<br/>在仿真中跟踪参考轨迹"]
    end
    CTRL --> OUT["物理合理的长时程人体行为<br/>滑动窗口实时生成"]
    subgraph DATA["GBC-100K 数据构建（训练各组件用）"]
        D1["约 500k 网络视频<br/>YOLOv8-Pose 过滤"] --> D2["TransNetV2 切镜头<br/>约 300k 视频片段"]
        D2 --> D3["TRAM 估计 SMPL → PoseScript<br/>多模态 LLM 生成 BehaviorScript / MotionScript"]
    end
    D3 -. 训练数据 .-> TP
    D3 -. 训练数据 .-> MP
```

图依据：论文 §4 Methodology（含 §4.1、§4.2、Data Collection）/ Fig. 2–3

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/From_Motion_to_Behavior__Hierarchical_Modeling_of_Humanoid_Generative_Behavior_C` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是一条「已排期、未深读」的待读条目：它只标出"从动作到行为的分层建模"这一命题下还有这么一篇，具体结论要等读完。**

- 本页能确认的只有索引层信息——arXiv 2506.00043（2025.06）、分类 04_Loco-Manipulation_and_WBC 与计划中的笔记文件夹；标题指向"用分层建模把动作层抬升为可生成的行为控制"，层级如何划分、行为如何被生成与约束，页内一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为方法对比或工程选型的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_from-motion-to-behavior.md](../../sources/papers/humanoid_pnb_from-motion-to-behavior.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2506.00043)：已通读方法部分（实验与附录以略读为准）：§3 预备知识（§3.1 GBC、§3.2 TAMP）、§4 方法（问题形式化、模型架构、数据收集、§4.1 指令式行为规划、§4.2 层次化运动生成控制）；§5 实验与补充材料未读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
