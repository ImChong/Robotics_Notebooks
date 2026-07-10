---
type: entity
tags: [agibot, motion-control, bfm, whole-body-control]
status: complete
updated: 2026-06-26
related:
  - ../overview/agibot-june-2026-release-technology-map.md
  - ../overview/agibot-release-category-05-body-foundations.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ./agibot-agile.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
summary: "BFM-2 是智元公开的运动小脑运控基座：二阶段 Motion-Between 设计，强调动作间稳定过渡与动态任务闭环；与 awesome-bfm-papers 学术索引及 paper-bfm-* 实体不同物。"
---

# BFM-2（智元运控基座）

**BFM-2** 是智元在 [2026-06 发布地图](../overview/agibot-june-2026-release-technology-map.md) 中推出的 **运动小脑 / 运控基座模型**（公开视频：[B站 BV1ZzGe6oEmk](https://www.bilibili.com/video/BV1ZzGe6oEmk/)）。文内称其为 **二阶段 Motion-Between 运控基座**，可在静态、预设动作或随机输入等状态下完成 **高稳定性动作插值与动态任务闭环**。

## 一句话定义

**补「动作与动作之间」的身体底座**——单个动作能做出不等于系统能连续运行；BFM-2 强调过渡、插值与肌肉记忆式闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 行为基础模型，可复用身体能力接口 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 为什么重要

- **能力底座位：** 与五个开源项目并列的 **两个能力底座之一**（另一为 [AGILE](./agibot-agile.md)）。
- **小脑叙事：** 高层规划完整时，身体层仍可能在 **动作衔接、随机扰动、动态闭环** 上失败。
- **产业语境：** [BFM 41 篇地图](../overview/bfm-41-papers-technology-map.md) 已记录智元把 **BFM-2** 推为运控基座并预告 BFM-3；本页补 **同发布会七段链路** 中的落点。

## 命名辨析（必读）

| 名称 | 含义 |
|------|------|
| **BFM-2（本页）** | 智元 **产品/基座** 代号 |
| [paper-bfm-*](./paper-bfm-22-phc.md) 等 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **学术论文索引实体** |
| [行为基础模型](../concepts/behavior-foundation-model.md) | 跨机构 **概念** 页 |

## 信息边界

- 截至 ingest 时，公开材料主要为 **视频标题与简介**（无字幕全文）；技术细节须以后续论文/仓库为准。

## 关联页面

- [身体能力底座分类 hub](../overview/agibot-release-category-05-body-foundations.md)
- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)

## 推荐继续阅读

- [机器人论文阅读笔记：TWIST2](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TWIST2__Scalable_Portable_and_Holistic_Humanoid_Data_Collection_System/TWIST2__Scalable_Portable_and_Holistic_Humanoid_Data_Collection_System.html)
- [机器人论文阅读笔记：SENTINEL](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SENTINEL__A_Fully_End-to-End_Language-Action_Model_for_Humanoid_Whole_Body_Control/SENTINEL__A_Fully_End-to-End_Language-Action_Model_for_Humanoid_Whole_Body_Control.html)
- [机器人论文阅读笔记：Agility Meets Stability](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Agility_Meets_Stability__Versatile_Humanoid_Control_with_Heterogeneous_Data/Agility_Meets_Stability__Versatile_Humanoid_Control_with_Heterogeneous_Data.html)
- [BFM-2 公开视频](https://www.bilibili.com/video/BV1ZzGe6oEmk/)
