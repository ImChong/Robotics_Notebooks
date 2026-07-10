---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2212.02500"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_physdiff.md
summary: "去噪扩散模型在人体动作生成上效果很好，但现有动作扩散模型忽视物理定律，常生成带明显伪影的动作——漂浮（floating）、脚滑（foot sliding/skating）、地面穿插（ground penetration）等。PhysDiff 把物理约束注入扩散过程：提出一个物理引导的动作投影（physics-guided motion projection）模块——在扩散的去噪步中，借物理仿真器里的动作模仿（motion imitation），把当前扩散出的（含噪）动作投影成一个物理可行的动作，再用它引导下一步去噪。如此生成的动作物理可信、自然，大幅减少上述伪影，在大规模人体动作数据集上取得SOTA 的动作质量与物理可信度。ICCV 2023 Oral。"
---

# PhysDiff

**PhysDiff: Physics-Guided Human Motion Diffusion Model** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

去噪扩散模型在人体动作生成上效果很好，但现有动作扩散模型忽视物理定律，常生成带明显伪影的动作——漂浮（floating）、脚滑（foot sliding/skating）、地面穿插（ground penetration）等。PhysDiff 把物理约束注入扩散过程：提出一个物理引导的动作投影（physics-guided motion projection）模块——在扩散的去噪步中，借物理仿真器里的动作模仿（motion imitation），把当前扩散出的（含噪）动作投影成一个物理可行的动作，再用它引导下一步去噪。如此生成的动作物理可信、自然，大幅减少上述伪影，在大规模人体动作数据集上取得SOTA 的动作质量与物理可信度。ICCV 2023 Oral。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| PhysDiff | 物理引导的动作扩散模型 |
| Motion Projection | 物理引导动作投影 |
| Motion Imitation | 物理仿真器里的动作模仿 |
| Floating / Foot Sliding | 漂浮 / 脚滑（被消除的伪影） |
| Ground Penetration | 地面穿插 |
| Denoising Step | 扩散去噪步 |

## 为什么重要

- **"生成 + 物理投影"是把生成动作变可执行的关键范式**，与 SafeFlow（物理引导整流流）、Heracles 等"物理可执行生成"一脉相承；
- **用物理仿真器的动作模仿做投影**，直接把"可被机器人执行"注入生成；
- 去脚滑/穿地正是人形动作重定向/跟踪要解决的；
- 物理可信动作可作人形参考运动，减少 sim-to-real 风险。

## 解决什么问题

动作扩散模型**不懂物理**： - 生成动作有**漂浮、脚滑、穿地**等伪影； - 纯数据驱动**无物理约束**，难保证可信。

PhysDiff 要：把**物理**注入扩散，生成**物理可信、少伪影**的动作。

## 核心机制

1. **把物理注入动作扩散**：物理引导动作投影模块；
2. **去噪步内物理投影 + 引导**：用仿真器动作模仿拉回可行流形；
3. **大减伪影**：漂浮/脚滑/穿地；
4. **SOTA 物理可信度**：大规模动作数据集（ICCV 2023 Oral）。

方法拆解（深读笔记小节）：物理引导动作投影（核心）；用投影结果引导下一步去噪；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model.html> |
| arXiv | <https://arxiv.org/abs/2212.02500> |
| 作者 | Ye Yuan、Jiaman Li、Yang Zou、Xiaolong Wang、Umar Iqbal、Sifei Liu、Jan Kautz（NVIDIA / Stanford） |
| 发表 | 2022 年 12 月 |
| 会议 | ICCV 2023（Oral） |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_physdiff.md](../../sources/papers/humanoid_pnb_physdiff.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model.html>
- 论文：<https://arxiv.org/abs/2212.02500>

## 推荐继续阅读

- [机器人论文阅读笔记：PhysDiff](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model.html)
