---
type: entity
tags: [company, scaling, world-action-model, video-policy, pretraining, evaluation, rhoda-ai]
title: Rhoda：网络视频预训练的真机 scaling 研究
status: complete
updated: 2026-10-16
summary: "Rhoda AI 博客（2026-09-10）在同一工业拆包任务上固定逆动力学模型与后训练流程，只改视频模型规模与预训练算力：规模 XS→S→M→L 的 100 秒内无人工干预完成率为 3.7%→65.0%→75.3%→84.7%，算力越多越好但数据充足时差距小；自定义 DINO FD 与任务表现同序；超过 200 小时真机评测，单任务单本体，闭源。"
related:
  - ./rhoda-ai.md
  - ./rhoda-direct-video-action.md
  - ./rhoda-dva-tactile-sensing.md
  - ../concepts/embodied-scaling-laws.md
  - ../concepts/world-action-models.md
  - ./generalist-gen0.md
  - ./dyna-2.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/sites/rhoda-ai.md
---

# Rhoda：网络视频预训练的真机 scaling 研究

| 字段 | 内容 |
|------|------|
| **机构** | Rhoda AI |
| **类型** | 产业官方博客（非 peer-reviewed 论文，无 arXiv） |
| **问题** | 扩大视频模型规模与预训练算力，能否在真实工业任务上提高机器人表现 |
| **发布** | 2026-09-10 |
| **开源** | **确认未开源**（2026-10-16 核查：无代码 / 权重 / 数据集链接） |

## 一句话定义

在 [DVA](./rhoda-direct-video-action.md) 框架下，用**通用网络视频**（不含动作）预训练因果视频模型，再用同一任务的机器人演示后训练；固定逆动力学模型、任务与评测，只变**参数量**与**预训练算力**，观察真机任务完成率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DVA | Direct Video-Action | 视频模型直接作策略，见 [DVA](./rhoda-direct-video-action.md) |
| DINO FD | DINO Fréchet Distance | 冻结 DINOv2 ViT-L/14 编码预测帧与真实帧后算特征分布 Fréchet 距离，越低越好 |
| IDM | Inverse Dynamics Model | 逆动力学模型；所有实验中保持固定 |
| XS / S / M / L | Extra-Small / Small / Medium / Large | 四档视频模型规模，同一预训练配方 |

## 为什么重要

- **少见的真机 scaling 曲线：** 多数 scaling 论述来自仿真或离线指标；这里每个策略做 100 到数百次真机试验，总计超过 **200 小时**真机评测。
- **回答「网络视频值不值」：** 把预训练规模与任务表现直接挂钩，可与 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md) 中以机器人交互数据为轴的结论对照。
- **提出可在预训练阶段监控的代理指标：** DINO FD 不需要机器人，也不需要后训练，便宜到可在预训练全程记录。

## 实验设置

- **任务：** 工业拆包——拆解轴承包装并分拣包装废料，14 个步骤、6 个子任务；每箱 10 kg。
- **指标：** at-speed completion rate——**100 秒内**、无人工干预完成全部目标才算成功；时限由作者设定，比客户要求更严，用来避免分数饱和。
- **控制变量：** 同一逆动力学模型（在更广的多机器人数据上训练一次）、同一任务、后训练流程、评测协议与推理频率。

## 实验结果（博客自报）

### 1. 模型规模（同一预训练配方）

| 规模 | 完成率 |
|------|--------|
| XS | 3.7% |
| S | 65.0% |
| M | 75.3% |
| L | 84.7% |

作者称 L 上趋势仍未饱和，预测更大模型会更高。

### 2. 预训练算力（M 模型四档）

| 相对算力 | 完成率 |
|----------|--------|
| 0.08× | 57.8% |
| 0.18× | 67.3% |
| 0.37× | 74.5% |
| 1× | 75.3% |

后训练数据充足时，三档较高算力差在 8 个百分点内；数据减少时差距拉开（25% 数据时 1× 为 60.2%，0.18× 为 29.1%），再减到 12.5% 时差距又缩小。交互项检验 p = 0.07，不显著。

### 3. 预训练质量 vs 任务表现

- 7 个检查点按 DINO FD 排序，任务表现基本同序，L 比下一名高约 10 个百分点。
- 同一次 M 模型后训练运行里，两个检查点的完成率可相差 **28 个百分点**；作者认为验证损失是糟糕的检查点选择器（与他人报告一致，自家实验同此结论），后训练时改用不需要机器人试验的代理指标选检查点。

## 与其他工作对比

页面**没有**与其他方法做直接性能对比，只在同一框架内比规模与算力。页面外链列出了 Generalist [GEN-0](./generalist-gen0.md)、Dyna [Dyna-2](./dyna-2.md) 与 NVIDIA EgoScale 等同样讨论 scaling 的工作；本页不判断数字高低，各家任务、数据与指标不同。

## 工程实践

| 项 | 实践要点 |
|----|----------|
| 选检查点 | 不要只看验证损失；预训练阶段可用留出网络视频上的 DINO FD 排序检查点（作者的相关性结论，非因果） |
| 数据 vs 算力 | 机器人数据充足时预训练算力的边际收益小；数据稀缺时预训练更重要 |
| 评测协议 | 设置严格时限避免分数饱和；公开试验数与置信区间，避免小差异被当成结论 |
| 开源状态 | **不适用源码运行时序图**——确认未开源 |

## 局限与风险

作者自列的局限：

- 真机试验数量有限（每策略 100 到数百次），小差异难以检出。
- 单任务、单本体、单一设置。
- 每个条件只做一次后训练，区间只反映试验间变异，不含训练种子间变异。
- 模型规模实验**未做算力匹配**，规模增益与训练 token 数、算力混在一起。
- 检查点选择与 DINO FD 的关系仅是 7 个检查点上的相关性，未证明可稳定选出最优检查点。
- 100 秒时限比客户要求严，绝对数值与策略间差距可能随之改变。

另需注意：数字均为公司自报，无第三方复现。

## 关联页面

- [Rhoda AI（公司入口）](./rhoda-ai.md)
- [DVA：因果视频模型作机器人策略](./rhoda-direct-video-action.md)
- [触觉 + DVA](./rhoda-dva-tactile-sensing.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [世界动作模型](../concepts/world-action-models.md)
- [GEN-0](./generalist-gen0.md) — 以真实交互数据为轴的 scaling 叙事
- [Dyna-2](./dyna-2.md) — 以人视频为预训练轴的闭源 WAM
- [具身大模型评测基准选型闭环知识链](../queries/embodied-eval-benchmark-selection-loop.md) — 本页 scaling 曲线属其 ③ 策略任务成功率评测层，且为博客自报的真机内部评测：复用结论前先按 ④ sim↔real / 可复现性层校准。

## 参考来源

- [Rhoda AI 官网研究页、新闻稿与团队页核查（2026-10-16）](../../sources/sites/rhoda-ai.md)
- 原文：<https://www.rhoda.ai/research/scaling-web-video-pretraining>

## 推荐继续阅读

- [Does Scaling Web-Video Pre-training Help Real Robots Do Real Work?](https://www.rhoda.ai/research/scaling-web-video-pretraining) — 官方博文，含附录 DINO FD 计算细节
- Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models* — <https://arxiv.org/abs/2001.08361>，页面引用的语言模型 scaling 参照
