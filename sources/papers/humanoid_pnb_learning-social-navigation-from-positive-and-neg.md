# Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记）

- **标题：** Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2510.12215>
- **项目页：** <https://chanwookim971024.github.io/PioneeR/>
- **代码：** **未开源**（2026-11-21 核查 `chanwookim971024/PioneeR`：仓库仅含项目页 HTML/静态资源，无代码、权重或数据下载）
- **入库日期：** 2026-07-10
- **通读：** 已完成
- **一句话说明：** 移动机器人在动态人群环境导航，需要策略既能适应多样人类行为、又遵守安全约束。本文基于 72 束 LiDAR（按 `g=3` 分组为 24 维 + 2 最近障碍距离角 + 目标方位角，`o ∈ R^29`）从正示范与负示范（`γ ∈ {-1, +1}` 平滑杠杆核 RKHS 密度奖励）学习偏好，并叠加到达与避障规则；10 Hz / 3 s前瞻控制器（`Δt=0.3 s`）通过 DAgger 蒸馏为 10 分量 MDN 学生以输出 aleatoric 与 epistemic 不确定性；Schindler 6000 电梯共乘仿真中 teacher SR 99.4–99.6%、MDN SR 98.0–100.0%。

## 原文通读摘录（arXiv:2510.12215v1 §I–§VI）

- **观测与动作空间：** 单轮平面运动模型 `u = [v, ω]^T`；观测向量 `o ∈ R^29`（24 组最小距离特征 + 2 个最近障碍距离/方位角 + 相对目标角 `φ_g`）。
- **奖励与前瞻 teacher：** 正示范 `γ=+1` 与负示范 `γ=-1` 在 RKHS 上用平滑杠杆核构造密度奖励，按剩余距离比自适应调制权重 `α_den = 1 + cos(π(1-r))`、`α_goal = 2(1-r)`、`α_obs = 1`；10 Hz 采样前瞻 `T=3 s`（120 个离散 `[v,ω]` 候选 + `α_EMA=0.5`）。
- **电梯共乘仿真实验：** 仿 Schindler 6000（1.8 m 门宽、2.5×2.7 m 轿厢、社会力行人半径 0.6 m、`v_pref ∈ {0.5,0.6,0.7}` m/s）；HR-RL（732 正 / 170 负）与 HL-RR（662 正 / 194 负）上 PioneeR teacher SR **99.4% / 99.6%**（CVaR-BF 72.8% / 71.4%，CrowdNav++ 78.6% / 64.8%）。
- **不确定性感知 MDN 蒸馏（Table III–IV）：** DAgger（50 条初始 + 10 轮×50 条）训得 10-component MDN 学生，HR-RL/HL-RR SR **98.0% / 100.0%**（高于 MLP 95.8% / 98.8%）；危险帧（人机距 `<0.8 m`）平均 epistemic（0.382/0.523 vs 0.290/0.342）与 aleatoric（0.407/0.426 vs 0.310/0.270）显著升高，可作降速与回退信号。
- 知识归纳见 wiki 实体页：[paper-notebook-learning-social-navigation-from-positive-and-neg](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md).

## 对 wiki 的映射

- [paper-notebook-learning-social-navigation-from-positive-and-neg](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)
- [PioneeR 项目页归档](../sites/pioneer-social-navigation.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html>
- 论文：<https://arxiv.org/abs/2510.12215>
