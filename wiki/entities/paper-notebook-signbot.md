---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2505.24266"
venue: "2025.05"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_signbot.md
summary: "SignBot：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# SignBot

**SignBot: Learning Human-to-Humanoid Sign Language Interaction** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

SignBot 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    CAM["摄像头观察手语使用者<br/>存为视频输入"] --> TR
    subgraph CB["大脑 Cerebral：手语推理（§IV-C）"]
        TR["手语翻译器<br/>LLM，视频特征对齐到语言模型维度<br/>用 CSL-Daily / How2Sign 文本微调"] --> RS["手语应答器<br/>DeepSeek-Chat API + 提示模板<br/>限定在 CSL 数据集词汇范围内"]
        RS --> GEN["手语生成器<br/>多语言 LM 编码文本 + 词级手语 token<br/>三个解码头自回归预测 UB / LH / RH 运动 token"]
        VQ["解耦 VQ-VAE 分词器<br/>上身 UB / 左手 LH / 右手 RH 各一个码本"] -.-> GEN
        GEN --> REC["重建为 SMPL-X 手语运动<br/>作为模仿目标"]
    end
    subgraph RT["运动重定向（§IV-A）"]
        V["手语视频 mesh"] --> BR["身体重定向<br/>双 T-pose 对齐，腕/肩增加自由度"]
        V --> HR["手部重定向<br/>映射到 Linker 手关节<br/>优化关键点向量差 + 帧间平滑正则"]
    end
    V -. 与文本标签一起训练 .-> GEN
    REC --> GOAL["模仿目标 o^y<br/>14 个参考关键点、参考关节位置与速度"]
    BR --> GOAL
    HR --> GOAL
    subgraph CE["小脑 Cerebellum：控制策略训练（§IV-B）"]
        GOAL --> UP["上身策略 π_upper<br/>跟踪重定向后的上身动作"]
        GOAL --> LOW["下身策略 π_lower<br/>保持平衡并适应上身动作<br/>腿式 H1 用 PPO，轮式 W1 用 MPC"]
        UP --> ACT["全身动作 a = 下身 + 上身<br/>IsaacGym 中域随机化训练"]
        LOW --> ACT
    end
    ACT --> RK["Ruckig 在线轨迹生成<br/>三阶（jerk）约束，避开关节极限"]
    RK --> ROBOT["真机执行手语<br/>H1 / W1 + Linker 灵巧手"]
```

图依据：论文 §IV-A / §IV-B / §IV-C / Fig. 2

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/SignBot__Learning_Human-to-Humanoid_Sign_Language_Interaction` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 SignBot 的待读条目；目前可确认的主要是一个归类判断——「人到人形的手语交互」被放进 04_Loco-Manipulation_and_WBC，而不是纯感知或语言类目。**

- 标题界定的问题是双向的手语交互，即以表达为目的的上肢/全身动作，这解释了它落在全身控制类目下的原因。
- 深读笔记尚未撰写、「实验与评测」明确留空，本页不可作为任何量化指标或实机结论的出处。
- 核心信息表未给 arXiv 行，深读状态与计划文件夹以 progress.json 为准；上溯路径为 [04_Loco-Manipulation_and_WBC 分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) 与 [总索引](../overview/humanoid-paper-notebooks-index.md)。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_signbot.md](../../sources/papers/humanoid_pnb_signbot.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2505.24266)：已通读方法部分（实验与附录以略读为准）：§III 问题定义、§IV SignBot（§IV-A 运动重定向、§IV-B 小脑控制策略训练、§IV-C 大脑手语推理与 Sim-to-Real 部署）；§V 实验与附录略读


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
