---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2604.00416"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_egonav.md
summary: "EgoNav 把「导航」拆成一个与机器人本体无关的轨迹分布预测问题：用一个 46M 参数的扩散 UNet，在仅 5 小时人类行走数据上学会「给定过去轨迹 + 360° 第一视角视觉记忆，未来该往哪些方向走」的多模态轨迹分布；推理时用 DDIM+DDPM 混合采样做到实时（Jetson Thor 上 110 traj/s），再由 receding-horizon 控制器从分布里挑路；最终零样本迁移到 Unitree G1，在没见过的室内外环境里连续走 37.5 分钟 / 1137 米 / 96%+ 自主率，还自发涌现出「等门开、绕人群、避玻璃墙」等行为。"
---

# EgoNav

**EgoNav: Learning Humanoid Navigation from Human Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

EgoNav 把「导航」拆成一个与机器人本体无关的轨迹分布预测问题：用一个 46M 参数的扩散 UNet，在仅 5 小时人类行走数据上学会「给定过去轨迹 + 360° 第一视角视觉记忆，未来该往哪些方向走」的多模态轨迹分布；推理时用 DDIM+DDPM 混合采样做到实时（Jetson Thor 上 110 traj/s），再由 receding-horizon 控制器从分布里挑路；最终零样本迁移到 Unitree G1，在没见过的室内外环境里连续走 37.5 分钟 / 1137 米 / 96%+ 自主率，还自发涌现出「等门开、绕人群、避玻璃墙」等行为。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| EgoNav | Egocentric Navigation | 本文系统名，第一视角导航 |
| DDIM / DDPM | Denoising Diffusion (Implicit / Probabilistic) Models | 两类扩散采样，分别快/稳，本文混合用 |
| DINOv3 | self-distillation 视觉骨干 | 冻结，用来抽深度传感器看不到的外观语义特征 |
| VAE | Variational Autoencoder | 把 360° 视觉记忆编码成低维 embedding |
| Receding-horizon | 滚动时域控制 | 每步只执行预测轨迹的前一小段，再重规划 |
| Embodiment-agnostic | 具身无关 | 学到的先验不绑定某个机器人本体 |

## 为什么重要

- **人类数据规模化**：给「用便宜的人类第一视角数据训机器人导航」提供了一个干净、可复制的范式
- **规划-行走解耦**：导航先验只管「往哪走」，locomotion 控制器只管「怎么走」，两者可独立替换/升级
- **扩散策略实用化**：证明扩散轨迹生成可以在边缘算力上实时跑，推动扩散从仿真走向真机
- **与 H\ / NaVILA / FocusNav 的位置**：那些工作偏「看哪里 / 语义规划 / 局部注意力」；EgoNav 给的是**「下一步轨迹分布」**这一中间层，可作它们的执行落地

## 解决什么问题

人形机器人导航有两个老大难：

1. **机器人数据太贵**：要在真机上采足量的导航演示，既慢又危险，还难覆盖各种场景； 2. **「规划」和「行走」割裂**：传统做法是上层给 waypoint、下层 locomotion 跟踪，中间这层「给定我看到的世界，我下一步该往哪走」往往要为每个机器人单独训。

## 核心机制

1. **范式**：把导航重新定义为**具身无关的轨迹分布预测**，从而可以用**便宜的人类行走数据**替代昂贵的机器人数据。
2. **零机器人数据 / 零微调**：仅 5 小时人类数据训练，**零样本**迁移到 Unitree G1 真机长时自主行走。
3. **表示**：360° 视觉记忆（RGB+深度+语义）+ 冻结 DINOv3，兼顾几何与外观，能识别**深度传感器盲区**（玻璃墙）。
4. **实时扩散**：DDIM+DDPM 混合采样把扩散轨迹生成压到 10 步、实时可跑，解决扩散策略「慢」的老问题。
5. **涌现社交/交互行为**：等门、绕人群等行为**不是显式编码**，而是从人类数据先验里自然涌现。

方法拆解（深读笔记小节）：输入：360° 第一视角「视觉记忆」+ 过去轨迹；模型：扩散 UNet 一次性出「未来轨迹分布」；实时推理：DDIM + DDPM 混合采样；执行：receding-horizon 控制器选路；关键数字。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html> |
| arXiv | <https://arxiv.org/abs/2604.00416> |
| 发表 | 2026-04-01 (arXiv) |
| 项目主页 | [egonav.weizhuowang.com](https://egonav.weizhuowang.com/) |
| 源码 | 数据集与模型「Coming Soon」，截至当前未公开 |
| 笔记阅读日期 | 2026-06-17 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_egonav.md](../../sources/papers/humanoid_pnb_egonav.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html>
- 论文：<https://arxiv.org/abs/2604.00416>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoNav](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html)
