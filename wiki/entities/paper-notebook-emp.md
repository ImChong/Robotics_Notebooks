---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2507.15649"
venue: "2025.07"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_emp.md
summary: "EMP：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# EMP

**EMP: Executable Motion Prior for Humanoid Robot Standing Upper-body Motion Imitation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

EMP 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    H["人体动作 m_source<br/>关键节点位置 + 边特征"] --> RT
    subgraph S1["§III 动作重定向网络"]
        RT["图卷积 Encoder f_e → 变换网络 f_tf<br/>→ VQ codebook 层 → Decoder f_d"] --> G["上身关节目标 g_t = q_target ∈ R15<br/>双臂 7-DoF ×2 + 腰 1-DoF"]
    end
    G --> EMP
    ST["机器人状态 s_t"] --> EMP
    subgraph S2["§V Executable Motion Prior (EMP)"]
        EMP["VAE 结构 EMP Net<br/>状态编码 f_s + 目标编码 f_t → 融合网络 f_ψ → 潜变量 z → 解码器<br/>输出 ĝ_t = f_θ(s_t, g_t)"]
    end
    EMP -->|"可执行的上身目标 ĝ_t"| PD2["PD 控制器 → 上身关节"]
    EMP -->|"ĝ_t 作为目标条件"| RL
    subgraph S3["§IV 下肢 RL 策略 π_lower"]
        RL["目标条件 RL 策略<br/>输入最近 T=15 帧 [q_t, a_t-1, rpy_t, g_t]<br/>输出 12 维腿部关节位置目标"]
    end
    RL --> PD1["PD 控制器 → 下肢力矩 τ_t"]
    PD1 --> ROBOT["人形机器人（仿真 / 真机）"]
    PD2 --> ROBOT
    ROBOT -->|"本体状态"| ST
    subgraph TR["EMP 训练（§V-B）"]
        WM["World Model f_w<br/>预测 ŝ_t+1，用于梯度反传"] --> LOSS["损失：重构 / 朝向 / 自碰撞 / 质心 / 平滑 / 正则"]
    end
    LOSS -.-> EMP
```

图依据：论文 §III–§V / Fig.2、Fig.3、Eq.(1)–(12)

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/EMP__Executable_Motion_Prior_for_Humanoid_Robot_Standing_Upper-body_Motion_Imita` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**EMP 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2507.15649（2025.07）、分类 04_Loco-Manipulation_and_WBC 与计划中的笔记文件夹；标题指向「站立姿态下的上半身动作模仿」这一子问题，机制见「流程总览」，实验细节页内一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_emp.md](../../sources/papers/humanoid_pnb_emp.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2507.15649)：已通读方法部分（实验与附录以略读为准）：§III 动作重定向、§IV RL 控制策略、§V Executable Motion Prior（含训练损失与 World Model）；§VI 实验以略读为准

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
