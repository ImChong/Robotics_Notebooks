---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2307.15042"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_tedi.md
summary: "去噪扩散概率模型（DDPM）逐步、小增量地合成样本——这种渐进性是其关键。TEDi 把这种渐进性应用到运动序列，并扩展 DDPM 以实现随时间变化的去噪，从而把\"扩散时间轴\"与\"运动时间轴\"两条轴纠缠（entangle）起来。具体：维护一个运动缓冲区（motion buffer），里面是越靠后越噪的姿态序列，对其迭代去噪，自回归地生成任意长的帧序列。每个扩散步只推进运动时间轴、而扩散时间轴保持静止，于是干净帧从缓冲区滑出、末端追加新噪声向量，实现长时程动作合成，适用于角色动画等。"
---

# TEDi

**TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

去噪扩散概率模型（DDPM）逐步、小增量地合成样本——这种渐进性是其关键。TEDi 把这种渐进性应用到运动序列，并扩展 DDPM 以实现随时间变化的去噪，从而把"扩散时间轴"与"运动时间轴"两条轴纠缠（entangle）起来。具体：维护一个运动缓冲区（motion buffer），里面是越靠后越噪的姿态序列，对其迭代去噪，自回归地生成任意长的帧序列。每个扩散步只推进运动时间轴、而扩散时间轴保持静止，于是干净帧从缓冲区滑出、末端追加新噪声向量，实现长时程动作合成，适用于角色动画等。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| TEDi | Temporally-Entangled Diffusion |
| DDPM | 去噪扩散概率模型 |
| Motion Buffer | 运动缓冲区（越后越噪） |
| Temporal Axis | 运动时间轴 |
| Diffusion Time-axis | 扩散时间轴 |
| Auto-regressive | 自回归生成 |

## 为什么重要

- **"滑动缓冲 + 时间轴去噪"是长时程生成的优雅机制**，对人形长时程动作/规划有借鉴；
- **自回归续写**契合在线控制（边生成边执行），与 UniAct 的流式思路相通；
- 把"扩散步"与"时间步"解耦/纠缠是值得迁移的生成范式；
- 长时程稳定性是人形长任务的共性挑战。

## 解决什么问题

扩散做**长时程**动作合成难： - 一次性生成超长序列**代价高、不稳**； - 朴素自回归易**断裂/漂移**。

TEDi 要：把扩散的渐进性"搬到"运动时间轴，**滑动缓冲**式地生成任意长动作。

## 核心机制

1. **时间纠缠扩散 TEDi**：把渐进去噪搬到运动时间轴；
2. **运动缓冲区 + 滑动生成**：干净帧滑出、末端续噪；
3. **任意长自回归合成**：长时程稳定；
4. **角色动画应用**：长时程动作合成。

方法拆解（深读笔记小节）：两轴纠缠：运动时间轴 × 扩散时间轴；运动缓冲区 + 滑动生成；应用；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis.html> |
| arXiv | <https://arxiv.org/abs/2307.15042> |
| 作者 | Zihan Zhang、Richard Liu、Kfir Aberman、Rana Hanocka（芝加哥大学 / Google） |
| 发表 | 2023 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_tedi.md](../../sources/papers/humanoid_pnb_tedi.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis.html>
- 论文：<https://arxiv.org/abs/2307.15042>

## 推荐继续阅读

- [机器人论文阅读笔记：TEDi](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis.html)
