---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2305.12577"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_guided-motion-diffusion-for-controllable-human-m.md
summary: "去噪扩散在文本条件的人体动作合成上很有前景，但纳入空间约束（如预定运动轨迹与障碍物）仍难——而这对连接孤立动作与其周遭环境至关重要。GMD（Guided Motion Diffusion）把空间约束注入动作生成：① 提出有效的特征投影方案，操纵动作表示以增强空间信息与局部姿态的一致性；② 配一个新的插补公式（imputation formulation），使生成动作可靠遵循全局运动轨迹等空间约束；③ 针对稀疏空间约束（如稀疏关键帧）易在反向步骤中被忽略的问题，提出稠密引导（dense guidance），把稀疏信号转成更密的信号去引导生成。实验证明 GMD 在文本动作生成上显著超 SOTA，同时支持轨迹跟随与避障等空间控制。"
---

# Guided Motion Diffusion for Controllable Human Motion Synthesis

**Guided Motion Diffusion for Controllable Human Motion Synthesis** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

去噪扩散在文本条件的人体动作合成上很有前景，但纳入空间约束（如预定运动轨迹与障碍物）仍难——而这对连接孤立动作与其周遭环境至关重要。GMD（Guided Motion Diffusion）把空间约束注入动作生成：① 提出有效的特征投影方案，操纵动作表示以增强空间信息与局部姿态的一致性；② 配一个新的插补公式（imputation formulation），使生成动作可靠遵循全局运动轨迹等空间约束；③ 针对稀疏空间约束（如稀疏关键帧）易在反向步骤中被忽略的问题，提出稠密引导（dense guidance），把稀疏信号转成更密的信号去引导生成。实验证明 GMD 在文本动作生成上显著超 SOTA，同时支持轨迹跟随与避障等空间控制。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| GMD | Guided Motion Diffusion |
| Spatial Constraint | 空间约束（轨迹/障碍） |
| Feature Projection | 特征投影（增强空间-姿态一致性） |
| Imputation | 插补公式（约束遵循） |
| Dense Guidance | 稠密引导（稀疏→密集信号） |
| Trajectory Following | 轨迹跟随 |

## 为什么重要

- **空间约束注入**是把"环境/轨迹"接进生成式动作的关键，与人形导航/避障相关；
- **稠密引导救稀疏约束**是通用技巧，可借鉴到稀疏关键帧/目标控制；
- **特征投影增强一致性**有助于约束真正生效；
- 可控动作生成是人形"目标驱动动作"的上游方法。

## 解决什么问题

文本动作扩散难纳入**空间约束**： - 预定**轨迹/障碍**难融入生成； - **稀疏约束**（稀疏关键帧）在反向去噪中**易被忽略**； - 要把**孤立动作**与**环境**连接。

GMD 要：把空间约束**可靠注入**文本动作扩散，支持轨迹跟随/避障。

## 核心机制

1. **GMD 空间约束注入**：把轨迹/障碍纳入文本动作扩散；
2. **特征投影**：增强空间信息与局部姿态一致性；
3. **插补公式 + 稠密引导**：可靠遵循约束、救稀疏信号；
4. **超 SOTA + 空间可控**：轨迹跟随与避障。

方法拆解（深读笔记小节）：特征投影（空间-姿态一致性）；插补公式（遵循全局轨迹）；稠密引导（救稀疏约束）；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis.html> |
| arXiv | <https://arxiv.org/abs/2305.12577> |
| 作者 | Korrawe Karunratanakul、Konpat Preechakul、Supasorn Suwajanakorn、Siyu Tang（ETH Zürich / VISTEC） |
| 发表 | 2023 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_guided-motion-diffusion-for-controllable-human-m.md](../../sources/papers/humanoid_pnb_guided-motion-diffusion-for-controllable-human-m.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis.html>
- 论文：<https://arxiv.org/abs/2305.12577>

## 推荐继续阅读

- [机器人论文阅读笔记：Guided Motion Diffusion for Controllable Human Motion Synthesis](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis/Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis.html)
