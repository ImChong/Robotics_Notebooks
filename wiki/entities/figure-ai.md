---
type: entity
tags: [hardware, humanoid, industry, vla]
status: complete
updated: 2026-05-07
related:
  - ./humanoid-robot.md
  - ./1x-technologies.md
  - ../methods/vla.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/repos/figure-ai.md
summary: "Figure AI 是美国人形机器人公司，以 Figure 02 整机与自研 Helix 视觉-语言-动作（VLA）模型为核心，强调全身协同与端侧推理，代表「垂直整合具身 AI」的一条主流工程路线。"
---

# Figure AI

## 一句话定义

**Figure AI** 构建「全栈人形」：**Figure 系列硬件** + **Helix 系列 VLA 模型**，目标是在真实家庭与物流场景中完成语言条件下的全身操作与移动。

## 为什么重要

- **VLA 商业叙事样板**：Helix 将视觉、语言与高频动作策略打包为可部署管线，是观测「大模型如何接地到人形控制」的关键样本。
- **垂直整合主张**：公开表述中强调机器人 AI 不能长期完全外包（与云端通用模型的分工仍在演化），对 Sim2Real、端侧算力与数据闭环提出了硬约束。
- **生态对标**：常与 Tesla Optimus、1X NEO、Agility Digit 等并列，用来评估美国人形赛道的进度条。

## 产品与模型（归纳）

| 名称 | 类型 | 说明 |
|------|------|------|
| **Figure 02** | 全尺寸人形整机 | 面向落地场景的第二代平台（细节以官方规格为准） |
| **Helix / Helix 02** | VLA 家族 | upper-body → full-body 控制扩展见 Figure 官方新闻稿 |

## 常见误区或局限

- **合作关系变化快**：曾与 OpenAI 在模型侧合作的新闻较多；后续转向自研 Helix。**选型讨论应以最新官方博客为准**，媒体报道仅作时间线辅助。
- **演示 ≠ 量产能力**：语音指令、抓取未知物体等亮点多在受控或半受控场景验证。
- **学术可用性**：Figure 不是典型「科研开箱平台」，复现其完整栈依赖未公开的模型与数据。

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [VLA](../methods/vla.md)
- [1X Technologies](./1x-technologies.md)
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)

## 参考来源

- [Figure AI 原始资料](../../sources/repos/figure-ai.md)

## 推荐继续阅读

- [Figure AI News](https://www.figure.ai/news)
- [Helix 专题页](https://www.figure.ai/helix)
