---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2603.09170"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_zerowbc.md
summary: "ZeroWBC 把\"教人形机器人做事\"的重心从昂贵的真机遥操作搬到几乎免费的人类第一人称视频——它不靠重定向把人体动作硬塞进机器人，而是直接训练一个视触运动（visuomotor）策略，从第一视角图像 + 少量机载本体感觉，端到端地输出全身关节命令；\"Zero\"指的是大规模真机遥操作上的\"近零\"依赖，由此让机器人学到坐下、踢、迈步、伸手等多样的\"自然交互\"行为，而不是被现有方法卡死在僵硬的步态模板里。"
---

# ZeroWBC

**ZeroWBC: Learning Natural Visuomotor Humanoid Control Directly from Human Egocentric Video** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

ZeroWBC 把"教人形机器人做事"的重心从昂贵的真机遥操作搬到几乎免费的人类第一人称视频——它不靠重定向把人体动作硬塞进机器人，而是直接训练一个视触运动（visuomotor）策略，从第一视角图像 + 少量机载本体感觉，端到端地输出全身关节命令；"Zero"指的是大规模真机遥操作上的"近零"依赖，由此让机器人学到坐下、踢、迈步、伸手等多样的"自然交互"行为，而不是被现有方法卡死在僵硬的步态模板里。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video.html> |
| arXiv | <https://arxiv.org/abs/2603.09170> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_zerowbc.md](../../sources/papers/humanoid_pnb_zerowbc.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video.html>
- 论文：<https://arxiv.org/abs/2603.09170>

## 推荐继续阅读

- [机器人论文阅读笔记：ZeroWBC](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video/ZeroWBC__Learning_Natural_Visuomotor_Humanoid_Control_from_Egocentric_Video.html)
