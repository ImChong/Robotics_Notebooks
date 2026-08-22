# ADEPT（arXiv:2608.19182）

> 来源归档（ingest）

- **标题：** ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning
- **类型：** paper / dexterous-manipulation / rl-pretraining / sim2real / geometric-fabric / visuo-tactile
- **arXiv abs：** <https://arxiv.org/abs/2608.19182>
- **PDF：** <https://arxiv.org/pdf/2608.19182>
- **HTML：** <https://arxiv.org/html/2608.19182>
- **项目页：** <https://adept-dexterity.github.io/>
- **机构：** 英伟达（NVIDIA）；密歇根大学（University of Michigan Robotics）
- **作者：** Jayjun Lee、Jessica Yin、Asif Rana、Nicholas Blauch、Sam Mady、Mohak Bhardwaj、Nima Fazeli、Nathan Ratliff、Karl Van Wyk、Ankur Handa
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **训练栈：** 大规模 GPU 并行仿真 PPO + ADR + PBT；Geometric Fabric 低层；DAgger distillation
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.19182](https://arxiv.org/abs/2608.19182) | 论文与附录 |
| 项目页 | [adept-dexterity.github.io](https://adept-dexterity.github.io/) | 真机 rollout、primitive 画廊 |
| 基准 | FMB [17] | peg insertion；parallel-jaw pipeline 对照 |
| 触觉 | TacMap [29]、SaTA [10] | Flexiv–Sharpa visuo-tactile student |

## 开源状态（步骤 2.5，2026-08-22 复核）

- **宣称将开源 / 待发布：** 项目页 **Code → Coming soon**（`is-pending`）；截至 **2026-08-22** **无** GitHub / Hugging Face URL。
- **处理：** wiki 标待发布；`## 源码运行时序图` 标不适用。
- **项目页补充：** 16 primitive 清单与 per-stage 真机累积成功率见 [`sources/sites/adept-dexterity-github-io.md`](../sites/adept-dexterity-github-io.md)（数据来自 `method-figure.js` Table 4 可视化）。

## 摘要级要点

- **问题：** 高 DoF arm–hand 每任务 from-scratch RL 重复发现 reach/grasp/lift/reorient；naïve fine-tune 预训练策略会 rapid collapse。
- **ADEPT 管线：** (1) generic **object reposing** pre-train（16 primitives）→ (2) structured **post-train**（BC distillation + critic warm-up + conservative PPO）→ (3) teacher→student distillation（two-stage vision curriculum）→ (4) zero-shot real deploy。
- **Fabric：** full joint **Cspace** geometric fabric（相对 DextrAH 的 PCA 子空间），sim 与真机同一控制器。
- **样本量：** pre-train ~8B steps；post-train ~3B/task；from-scratch 单任务 ~9B 且常失败。
- **真机：** Kuka–Allegro RGB student FMB **5/10**（star）、**3/10**（square/round）；Flexiv–Sharpa visuo-tactile **8/10**；dish **6/10**；5–10 s/trial vs FMB pipeline 20–70 s。

## 核心摘录（面向 wiki 编译）

### 1) Post-training 三步骤（§3.3）

1. **BC actor distillation** — \(\pi_{pre}\) → \(\pi_{post}\)（扩展观测 40k iter）
2. **Critic warm-up** — 冻结 \(\pi_{post}\)，训 \(V_{post}\) ~20 PPO iter
3. **Conservative PPO** — actor LR 1e-5（decay from 1e-3），clip 0.05，critic LR 5e-5

消融：LR 1e-3 必 collapse；BC 减半 adaptation time；critic warm-up +17.6% SR。

### 2) Pre-train 泛化（Table 1）

| Embodiment | Primitive SR | FMB peg SR | VisDex SR |
|------------|--------------|------------|-----------|
| Kuka–Allegro | 0.73±0.003 | 0.76±0.003 | 0.77±0.011 |
| Flexiv–Sharpa | 0.64±0.007 | 0.58±0.011 | 0.61±0.015 |

### 3) Zero-shot reposing on FMB ADR path（Table 2 节选）

Kuka–Allegro：ADR 20 → **71.9%**；ADR 50（insertion contact）→ **0%**。Post-training 从 ADR 20 起步。

### 4) 真机 per-stage（Table 4 节选）

Flexiv–Sharpa visuo-tactile FMB square/round：Reach 10/10 → Insert **8/10**；vision-only Insert **3/10**。

## 对 wiki 的映射

- 沉淀实体页：[ADEPT](../../wiki/entities/paper-adept-dexterity.md)
- 交叉补强：[manipulation 任务](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Ego2Robot](../../wiki/entities/paper-ego2robot.md)（人类视频→机器人数据对照）

## 当前提炼状态

- [x] arXiv HTML 方法 / Table 1–4 / 消融摘录
- [x] 项目页开源核查：Code Coming soon
- [x] 升格 `wiki/entities/paper-adept-dexterity.md`
