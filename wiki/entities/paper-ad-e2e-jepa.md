---
type: entity
project_id: ad-e2e-jepa
tags: [paper, autonomous-driving, end-to-end-driving, jepa, world-models, latent-dynamics, planning, nyu, ami-labs]
status: complete
updated: 2026-10-11
arxiv: "2609.34085"
venue: "arXiv 2026"
project: https://github.com/HaoranZhuExplorer/AD-E2E-JEPA
code: https://github.com/HaoranZhuExplorer/AD-E2E-JEPA
related:
  - ./paper-lewm.md
  - ./paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md
  - ../concepts/functional-taxonomy-world-models.md
sources:
  - ../../sources/papers/ad_e2e_jepa_arxiv_2609_34085.md
  - ../../sources/repos/ad-e2e-jepa.md
summary: "面向端到端自动驾驶的动作条件 JEPA 世界模型：冻结 DINOv3 ViT-L，以带 SIGReg 的可学习投影器压缩 patch 表征，在 NAVSIMv2 上做无策略训练的目标条件零样本规划；官方仓库暂未发布可运行源码。"
---

# AD-E2E-JEPA：端到端自动驾驶的联合嵌入预测架构

## 一句话定义

> AD-E2E-JEPA 冻结 DINOv3 视觉骨干，以带 SIGReg 正则的可学习投影器压缩未来表征，再通过动作条件潜空间 rollout 从轨迹词表中选择最接近目标观测的驾驶轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AD-E2E-JEPA | Autonomous Driving with an End-to-End Joint-Embedding Predictive Architecture | 本文自动驾驶世界模型与规划框架 |
| JEPA | Joint-Embedding Predictive Architecture | 在表征空间预测目标观测嵌入的架构 |
| SIGReg | Sketched Isotropic Gaussian Regularizer | 以随机投影正态性检验鼓励嵌入各向同性，降低表示坍塌风险 |
| E2EAD | End-to-End Autonomous Driving | 端到端自动驾驶 |
| NAVSIMv2 | Navigation Simulation Benchmark v2 | 本文采用的驾驶规划评测基准版本 |
| EPDMS | Ego Progress, Driving Metrics and Safety | NAVSIM 汇总评分，包含乘法安全项 |
| FDE | Final Displacement Error | 所选轨迹终点与目标真值位置间的平均距离 |

## 为什么重要

端到端驾驶常以模仿学习直接拟合示范轨迹，难以分离「世界模型的预测质量」与「策略的示范拟合质量」。本文把驾驶动作作为世界模型条件输入，在不训练驾驶策略的设置下，用真实未来图像作为目标，让模型比较候选轨迹对应的未来潜状态。它由此直接检验 JEPA 表征是否支持规划，而不只是支持视觉特征学习。

核心工程贡献并非新的视觉骨干，而是把冻结视觉模型的 dense patch 表征压到适合多候选 rollout 的尺寸，并用 SIGReg 保持压缩表征的信息性。它与 [LeWM](./paper-lewm.md) 的像素端到端小模型路线、[DINO-WM](./paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) 的预训练视觉特征规划形成互补参照。

## 核心信息

| 项 | 内容 |
|----|------|
| 作者 | Haoran Zhu、Wancong Zhang、Yann LeCun、Anna Choromanska |
| 机构 | 纽约大学（NYU）；先进机器智能实验室（AMI Labs） |
| 论文 | arXiv:2609.34085，2026-09-28 提交 |
| 输入 | 前视相机图像历史与相对自车位姿 |
| 视觉编码 | 冻结 DINOv3 ViT-L |
| 预测器 | AdaLN 风格 Transformer，使用 RoPE |
| 规划 | 对候选轨迹做潜状态 rollout，按终点潜状态与目标图像嵌入的距离选择 |
| 官方代码 | 仓库存在，但截至 2026-10-11 仍未公开可运行实现；状态见[仓库核查](../../sources/repos/ad-e2e-jepa.md) |

## 核心原理

### 动作条件潜空间动力学

对前视图像提取冻结 DINOv3 patch 表征。每步动作是相邻帧间的相对自车位姿变化 \(a_t=[\Delta x,\Delta y,\Delta\theta]\)。历史嵌入和动作嵌入输入 AdaLN predictor，预测下一个时刻的目标嵌入。基础监督是预测 embedding 与冻结目标 embedding 间的 MSE，不要求像素重建，也不把专家轨迹作为策略标签训练驾驶策略。

### 压缩 patch 与 SIGReg

可学习投影器由两层步长为 2 的卷积构成，共享用于历史帧与未来目标帧：空间 patch 数降为原来的 1/16，通道维从 1024 降为 256（1/4）。投影目标端使用 stop-gradient，训练损失由投影空间预测 MSE 与加权 SIGReg 组成。

SIGReg 在每个时间点、每个 patch 位置分别沿 batch 维施加，以随机一维投影上的 Epps–Pulley 检验逼近各向同性高斯分布。作者给出的默认配置为 17 个 knots、1024 个随机方向。该正则用于降低 projector 映射坍塌风险，并不等同于安全约束或碰撞代价。

### 训练与规划流程

```mermaid
flowchart LR
  Obs["历史图像与自车位姿"] --> Enc["冻结 DINOv3 ViT-L"]
  Enc --> Proj["共享可学习投影器"]
  Proj --> Pred["动作条件 AdaLN predictor"]
  Act["候选相对位姿序列"] --> Pred
  Pred --> Roll["潜空间 rollout"]
  Goal["目标未来图像"] --> GoalEnc["DINOv3 与共享投影器"]
  GoalEnc --> Score["计算预测与目标嵌入距离"]
  Roll --> Score
  Score --> Select["选择代价最低的轨迹"]
```

训练可仅预测下一未来帧，也可加入覆盖完整未来时域的 teacher-forcing 与 rollout consistency loss。规划时，目标为未来帧图像；模型对候选动作序列自回归 rollout，最小化预测终点 embedding 与目标 embedding 的平方距离。候选取自 8192 条 NAVSIM 驾驶轨迹锚点，作者按角度均匀下采样（例如 256 条）以平衡速度和轨迹粒度。

这个范式不训练驾驶策略，但使用了目标未来观测与预定义轨迹词表。因此它是目标条件零样本规划实验，不等同于无需目标先验的通用自动驾驶系统。

## 评测与结果

- **设置：** NAVSIM 数据集与 NAVSIMv2 navtest-stage-1。基准比较使用约 10 小时、2 Hz 的 navtrain 视频；扩展训练使用 trainval 训练部分约 70 小时数据。历史为 4 帧（2 秒），未来预测 8 帧（4 秒）。
- **时延：** 256 候选、单场景约 0.8 秒；DINO-WM / JEPA-WM 在 100 场景对比设置下约 91.8 / 101.0 秒，论文据此报告约 100 倍加速。计时使用 NVIDIA A100 80 GB 并行评估候选。
- **全量测试：** trainval + rollout、256 候选在 12,146 个测试场景上得到 EPDMS 67.3、EPDMS† 84.1、FDE 4.0 m、Top-1/Top-5 命中率 53.8% / 82.7%。
- **候选精度—时延交换：** 同一训练配置扩至 8192 候选，EPDMS / EPDMS† 为 72.9 / 86.5，FDE 2.8 m、平均航向误差 2.0°；时延增至 18.2 秒/场景。因此最高分与 0.8 秒的工作点不能合并描述。
- **表征迁移：** 单视角、4 帧 NAVSIMv2 设置下，预训练 projector 将下游模仿学习 EPDMS 从随机初始化 projector 的 80.2 提升到 85.4。该实验检验的是 projector 迁移，不是零样本规划成绩。
- **比较边界：** DINO-WM/JEPA-WM 只在 100 场景子集作昂贵比较；AD-E2E-JEPA 另报告 12,146 场景全量结果。不同子集的分数不能直接排序。

## 结论

**总判：AD-E2E-JEPA 通过 SIGReg 正则化的压缩 projector 显著降低冻结视觉世界模型的规划成本，并显示自监督投影表征可迁移到模仿学习；现有证据仍是 NAVSIMv2 目标条件规划，而非实车安全闭环。**

1. 复现或比较时同时报告候选数、GPU、场景数与训练 split；256 候选 0.8 秒和 8192 候选 72.9 EPDMS 是不同工作点。
2. 把 EPDMS 与 EPDMS† 一起读；后者移除了乘法安全项，不代表安全能力。
3. 单独跟踪 rollout loss 对 FDE 和 hit rate 的作用，不要只看汇总分。
4. 区分 100 场景对照与 12,146 场景全量测试，避免跨子集排名。
5. 当前仓库没有可运行训练、规划或部署入口；源码发布后再核验数据管线、依赖、权重与报告数字。
6. 真实部署还需显式安全过滤、闭环道路测试和传感器变化验证。

## 源码运行时序图

**不适用（截至 2026-10-11）。** 官方仓库只有 README 和 overview 图片，README 表示源码仍在准备、预计 2026-10-19 前发布。当前没有可核对的训练、推理、数据加载或部署入口，因此不绘制猜测性的代码时序图；详见[官方源码归档](../../sources/repos/ad-e2e-jepa.md)。

## 工程实践

- **对齐输入：** 图像与自车位姿历史共同定义上下文；动作使用相邻帧相对位姿变化，坐标参照必须一致。
- **监控压缩：** 分别记录投影前后 token 数、通道维、显存与单位候选 rollout 延迟；16 倍 patch 减少与 4 倍通道压缩是两种不同维度。
- **区分损失功能：** MSE 训练预测，SIGReg 约束表征分布，rollout loss 改善多步一致性；它们不是碰撞风险损失。
- **分开评测工作点：** 提高候选数改善搜索粒度但延长时延；报告分数时必须附候选数和场景规模。
- **源码开放边界：** 论文 PDF 声明代码可用，但官方 README 仍标注准备公开；以仓库可访问实现为准。

## 局限与风险

- **目标可见性：** 规划实验把 ground-truth future image 当作目标，测试给定目标时如何挑轨迹，不测试如何自主决定目的地。
- **安全不是规划代价：** 潜空间终点距离是选轨迹的目标；安全通过 NAVSIM 指标评估，未在该潜在距离中显式优化。
- **离线伪仿真：** NAVSIMv2 结果不能替代真实车辆闭环、法规验证或故障安全测试。
- **观测范围：** 方法设定为前视相机与平面自车位姿，不能自动推广到多相机、激光雷达或其他接口。
- **吞吐依赖候选规模：** 从 256 到 8192 候选，分数提升但时延由约 0.8 秒升至 18.2 秒。
- **源码状态矛盾：** arXiv PDF 称 source code available，仓库 README 截至入库日称源码准备中；本文依当前仓库现况记录为尚不可复现。

## 与相关路线对比

| 工作 | 方法定位 | 与本文的关系 |
|------|----------|--------------|
| [LeWM](./paper-lewm.md) | 像素端到端训练轻量动作条件 JEPA | LeWM 强调小型可训练骨干；本文冻结 DINOv3 并压缩 patch 表征 |
| [DINO-WM](./paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) | 预训练视觉特征上的零样本潜空间规划 | 本文增加面向驾驶规划的 projector 与 SIGReg，报告 NAVSIMv2 指标 |
| [世界模型功能分类](../concepts/functional-taxonomy-world-models.md) | 按系统输出区分 Renderer / Simulator / Planner | 本文归属 Planner：潜在未来服务目标条件轨迹选择，而非生成像素视频 |

## 关联页面

- [LeWorldModel / LeWM](./paper-lewm.md) — 小型像素端到端 JEPA 对照路线
- [DINO-WM](./paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) — 冻结视觉特征的零样本规划基线
- [世界模型功能分类（Renderer / Simulator / Planner）](../concepts/functional-taxonomy-world-models.md) — 从系统功能定位本文的潜在规划器

## 参考来源

- [AD-E2E-JEPA 论文归档](../../sources/papers/ad_e2e_jepa_arxiv_2609_34085.md)
- [官方仓库状态归档](../../sources/repos/ad-e2e-jepa.md)

## 推荐继续阅读

- [arXiv:2609.34085](https://arxiv.org/abs/2609.34085)
- [官方 GitHub 仓库](https://github.com/HaoranZhuExplorer/AD-E2E-JEPA)
- [DINO-WM 论文](https://arxiv.org/abs/2411.04983)
