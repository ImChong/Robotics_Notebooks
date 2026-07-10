---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2606.14160"
related:
  - ../overview/paper-notebook-category-09-state-estimation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_gait.md
summary: "GAIT 把足式机器人的本体感知测量做成「惯性-腿部（Inertial-Leg, IL）分词」——IMU 是一路 token、每条腿各是一路 token——再用轻量的 Perceiver IO 交叉注意力 学习「哪路测量此刻更可信」。因为一条腿只有触地时的前向运动学速度才可靠，注意力天然学到了「按接触状态重新赋权」这件事，不需要显式接触估计器。网络只用 trot（对角小跑） 数据训练，却能泛化到 bound / pace / pronk 等未见步态，并把估出的机身速度喂进 IEKF 得到完整位姿——同时推理只需 0.12 MFLOPs、0.7 ms。"
---

# GAIT

**GAIT: Legged Robot Proprioceptive State Estimation with Attention over Inertial-Leg Tokens** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：09_State_Estimation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

GAIT 把足式机器人的本体感知测量做成「惯性-腿部（Inertial-Leg, IL）分词」——IMU 是一路 token、每条腿各是一路 token——再用轻量的 Perceiver IO 交叉注意力 学习「哪路测量此刻更可信」。因为一条腿只有触地时的前向运动学速度才可靠，注意力天然学到了「按接触状态重新赋权」这件事，不需要显式接触估计器。网络只用 trot（对角小跑） 数据训练，却能泛化到 bound / pace / pronk 等未见步态，并把估出的机身速度喂进 IEKF 得到完整位姿——同时推理只需 0.12 MFLOPs、0.7 ms。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| Proprioceptive | - | 本体感知（IMU + 关节编码器），不含视觉 / LiDAR |
| IL Token | Inertial-Leg Token | 把惯性测量与各条腿测量分别编码成独立 token |
| Perceiver IO | - | 用少量可学习 latent token 做交叉注意力，把复杂度从 O(N²) 降到 O(MN) |
| IEKF / InEKF | (Invariant) Extended Kalman Filter | 不变扩展卡尔曼滤波，状态定义在李群 SE₂(3) 上 |
| NMN | Neural Measurement Network | 对比基线：学习式测量网络，预测接触概率 + 机身速度 |
| GRU | Gated Recurrent Unit | 门控循环单元，这里用来编码时间维 |
| RMSE / ATE / RE | Root Mean Sq. Err / Absolute Traj. Err / Relative Err | 速度与轨迹误差指标 |
| Slip Rejection (SR) | - | 打滑异常剔除，接触辅助 IEKF 的增强版 |

## 为什么重要

- **省掉接触估计器**：接触辅助 IEKF 的老大难是「接触判断错 → 约束错」，GAIT 用注意力把接触信息隐式吸进权重，绕开了这条脆弱链路
- **结构化输入范式**：「按传感器语义分词」是比「拼大向量」更可迁移的输入设计，对人形（更多自由度、更多接触点）尤其有价值
- **学习 + 滤波的干净接口**：网络只负责出「速度 + 不确定度」，重活交给 IEKF，兼顾学习的表达力与滤波的几何一致性
- **实时友好**：亚毫秒推理 + 极低 FLOPs，可直接跑在机载算力受限的四足 / 人形上

## 解决什么问题

足式机器人估计机身速度 / 位姿时，主流有两条路，各有死结：

1. **接触辅助 IEKF（模型驱动）**：靠「支撑足零速度」假设做测量更新，但**接触判断一旦出错**（打滑、软地形、腾空步态）就会注入错误约束，需要额外的接触估计器 + 打滑剔除逻辑来打补丁； 2. **学习式估计（如 NMN）**：直接把所有传感器**拼成一个大向量**喂进网络。问题是网络看不到「测量的结构」——它不知道第 3 维来自哪条腿、这条腿此刻是否触地，泛化到训练时没见过的步态就容易崩。

## 核心机制

1. **Inertial-Leg 分词**：不再把测量拍平成大向量，而是保留「惯性 / 各条腿」的结构，让网络能按测量来源区别对待。
2. **接触隐式建模**：注意力自动学出「触地腿权重高、腾空腿权重低」，**省掉了显式接触估计器与打滑剔除逻辑**。
3. **跨步态泛化**：只用 trot 训练，就在 bound / pace / pronk 等未见步态上大幅超越 NMN。
4. **轻量高效**：0.12 MFLOPs / 步、0.7 ms 延迟，比标准自注意力低约 2.8×、比 NMN 低约 3.3×，适合真机实时部署。

方法拆解（深读笔记小节）：Inertial-Leg（IL）分词；Perceiver IO 交叉注意力；GRU-MLP 时间编码 + 不确定度输出；两段式损失；接进 IEKF 做完整状态。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 09_State_Estimation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia.html> |
| arXiv | <https://arxiv.org/abs/2606.14160> |
| 发表 | 2026-06-12 (arXiv) |
| 源码 | 截至当前未见公开发布（论文未给出 GitHub / 项目页链接） |
| 笔记阅读日期 | 2026-07-04 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-09-state-estimation](../overview/paper-notebook-category-09-state-estimation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_gait.md](../../sources/papers/humanoid_pnb_gait.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia.html>
- 论文：<https://arxiv.org/abs/2606.14160>

## 推荐继续阅读

- [机器人论文阅读笔记：GAIT](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia.html)
