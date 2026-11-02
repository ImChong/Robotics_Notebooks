---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-10-06
arxiv: "2510.17792"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-09-compliance-contact.md
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_softmimic-learning-compliant-whole-body-control.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
summary: "SoftMimic：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# SoftMimic

**SoftMimic: Learning Compliant Whole-body Control from Examples** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：14_Human_Motion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

SoftMimic 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 在 [运动小脑 64 篇技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中归类为 **I 柔顺与接触**（60/64）：接触：柔顺全身控制也在成为 tracking 条件。
- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 笔记尚未写出；方法流程见下方「流程总览」（依据论文方法部分），实验结论以论文为准。

## 流程总览

```mermaid
flowchart TD
    subgraph OFF["离线：Compliant Motion Augmentation（§III-C）"]
        A["人体动作数据集（AMASS / LAFAN1 重定向）<br/>参考动作 q_ref"]
        B["采样交互事件：外力 wrench W_ext + 指令刚度 K_cmd"]
        C["可行性检查：IK 失败则缩小 wrench 重试，低于传感噪声下限则丢弃"]
        D["Differential IK 求解器（Mink + MuJoCo）<br/>优先级任务：Compliant Interaction → Foot Placement → CoM → Keypoint Posture → Joint Posture"]
        E["增强数据集 D_aug：(q_ref, w, K_cmd, q_aug)"]
        A --> B --> C --> D --> E
    end
    subgraph ON["在线：SoftMimic RL 训练（§III-B）"]
        F["采样动作片段 + 期望刚度 + Force Field 外力<br/>（对数均匀刚度采样、按速度采样碰撞事件）"]
        G["策略 π 观测：本体感知、基座状态、上一动作、q_ref、刚度指令（3 步历史）<br/>不直接观测外力"]
        H["输出关节位置目标 → PD 控制器"]
        I["奖励：与 q_aug 对齐的 DeepMimic 式跟踪奖励 + 平滑奖励 + 弹簧式柔顺奖励"]
        F --> G --> H --> I
    end
    E -->|"作为奖励目标、初始化与终止判据"| I
    I -->|"PPO 更新"| G
    H --> J["部署：单一策略 + 运行时刚度指令<br/>Unitree G1 真机"]
```

图依据：论文 §III-A 至 §III-D / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/14_Human_Motion/softmimic-learning-compliant-whole-body-control` |
| arXiv | <https://arxiv.org/abs/2510.17792> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前是双索引下的待读条目：SoftMimic 在运动小脑技术地图里已有明确定位，但论文自身的机制归纳仍待深读笔记补齐。**

- 已确认的定位是：[运动小脑 64 篇技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 将其归入 **I 柔顺与接触**（60/64），对应「柔顺全身控制也在成为 tracking 条件」这一趋势判断。
- 其余均为策展元数据：分类 14_Human_Motion、arXiv 2510.17792、深读笔记待撰写。
- 页面的作用是让你能按分类找到它；机制、实验与适用边界须等深读笔记完成后回填，此前不宜作为技术结论引用。

## 与其他页面的关系

- 分类页：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 关联页面

- [磁驱动双稳态软跳跃机器人](./paper-bistable-soft-jumper-magnetic.md)

## 参考来源

- [humanoid_pnb_softmimic-learning-compliant-whole-body-control.md](../../sources/papers/humanoid_pnb_softmimic-learning-compliant-whole-body-control.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2510.17792>
- [arXiv PDF](https://arxiv.org/pdf/2510.17792)：已通读方法部分（实验与附录以略读为准）：§I 引言、§II 相关工作、§III Method（§III-A–D）与 Fig.2（§IV 结果与附录仅略读）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
