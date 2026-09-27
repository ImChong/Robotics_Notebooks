---
type: entity
tags: [paper, manipulation, keypoints, diffusion-policy, generalization]
status: complete
updated: 2026-09-27
arxiv: "2609.28818"
related:
  - ../overview/embodied-research-12-papers-recover-wam-technology-map.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/keygen_arxiv_2609_28818.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md
summary: "KeyGen（2609.28818）：点云无监督 3D 关键点 + 物体中心几何条件化扩散策略，测类别内新实例泛化。"
---

# KeyGen

**KeyGen: Unsupervised Keypoint based Object-Centric Representations for Category-Level Policy Generalization**（[arXiv:2609.28818](https://arxiv.org/abs/2609.28818)，[项目页](https://robo-keygen.github.io/)）收录自 [具身智能小站 12 篇盘点（恢复/WAM 专题）](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)。

## 一句话定义

**从点云学规范化关键点表征，为 manipulation 策略提供 object-centric 条件以实现 category-level 泛化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SR | Success Rate | 任务成功率 |
| WAM | World Action Model | 联合预测未来观测与动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 纳入 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 与同期失败恢复、异步 WAM、接触感知、持续学习、安全 RL 条目横向对照。
- 步骤 2.5 开源结论：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28818](https://arxiv.org/abs/2609.28818) |
| **项目页** | https://robo-keygen.github.io/ |
| **代码** | — |
| **开源** | **待发布** |

## 实验与评测（公众号口径）

- 指标与数字以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md) 与 **原文 PDF** 为准；读复现前核对仿真/真机与 attempt 定义。


## 源码运行时序图

**不适用**（无统一官方入口或未开源）。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Diffusion Policy](./paper-diffusion-policy.md) | 动作 chunk 去噪的基线范式；KeyGen 在其条件端换成 **物体中心几何**（关键点）以追求类别内泛化 |
| [DP3](./painode-209-3ddiffusionpolicydp3.md) | 同为 **点云条件扩散策略**；DP3 直接用点云表征，KeyGen 先做 **无监督 3D 关键点 + 规范化**，把条件压成 object-centric 结构 |
| [SimToolReal](./paper-sa-2602-16863-simtoolreal-an-object-centric-policy-for-zero-sh.md) | 同属 **object-centric policy**：SimToolReal 面向灵巧工具的零样本 sim2real；KeyGen 面向 **category-level 新实例** 泛化 |
| [AnyBody](./paper-anybody-keypoint-humanoid-control.md) | 同为 **关键点条件化**，但 AnyBody 的关键点是 **人形身体稀疏关键点**（全身意图）；KeyGen 的关键点在 **被操作物体** 上 |
| [KnowBody](./paper-knowbody.md) | 同期「改条件化」条目：KnowBody 给冻结 VLM 补 **身体关系模型**；KeyGen 给策略补 **物体几何表征** |

## 结论

**总判：KeyGen 适合作为「从点云学规范化关键点表征，为 manipulation 策略提供 object-centric 条件以实现 catego…」方向的入口页；机制细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 对照，避免与其它 arXiv 混淆。
2. 开源为 **待发布** 时从项目页/GitHub 再核实一次再写复现计划。
3. 涉及异步 WAM 或恢复评测时，同时记录 **正常起点 SR** 与 **偏差后恢复率**（若适用）。

## 关联页面

- [具身研究 12 篇（恢复/WAM）技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/keygen_arxiv_2609_28818.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)

## 推荐继续阅读

- [arXiv:2609.28818](https://arxiv.org/abs/2609.28818)
- [项目页](https://robo-keygen.github.io/)

