# AutoIntervene（arXiv:2608.07065）

> 来源归档（ingest）

- **标题：** AutoIntervene: Calibrated Intervention for Action-Chunking Imitation Learning Policies
- **缩写：** **AutoIntervene**
- **类型：** paper / interactive-imitation / action-chunking / dagger / bimanual / deployment-monitor
- **arXiv：** <https://arxiv.org/abs/2608.07065>
- **HTML：** <https://arxiv.org/html/2608.07065>
- **PDF：** <https://arxiv.org/pdf/2608.07065>
- **项目页：** <https://aus.bot/AutoIntervene/>（用户指定；论文亦写 <https://aus.bot/research/autointervene/>）— 归档见 [`sources/sites/aus-bot-autointervene.md`](../sites/aus-bot-autointervene.md)
- **代码：** 截至 **2026-08-11** 项目页 **未列训练/推理仓**；GitHub `123qwedsa123/AutoIntervene` 仅为项目页静态镜像（HTML/CSS），**非**可运行实现
- **作者：** Jinhe Tang、Weiming Zhi\*（悉尼大学 / Australian Center For Robotics；Vanderbilt College of Connected Computing）
- **机构：** The University of Sydney；PAIR Lab（aus.bot）；Vanderbilt University
- **入库日期：** 2026-08-11
- **一句话说明：** 面向 **action-chunking** 策略的 **双向校准接管**：用成功轨迹的 visual-action memory 打视觉支持与动作风险分；phase-local 支持决定切到操作员，global 支持决定交回策略；干预片段作下一轮针对性监督。九项真机双臂任务，平均成功率与操作员时间优于人工接管 / 追加全演示。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2608.07065>
- **核心贡献：** Chunk 策略在分布外仍产出「平滑但错」的动作块。AutoIntervene 在线监控提议 chunk，相对成功参考检索打分，**自动** policy↔operator 切换；阈值由 held-out 专家演示分位数校准，避免手调 cutoff。成功 rollout 的干预段用于下一轮适配（选择性 DAgger）。
- **对 wiki 的映射：**
  - [AutoIntervene 实体](../../wiki/entities/paper-autointervene.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [DAgger](../../wiki/methods/dagger.md)

### 2) Visual-Action 支持与双向切换（§III）

- **链接：** Method
- **核心贡献：**
  - Query \(\mathcal{Q}_t=(E_t,A_t)\)：多相机 embedding + 提议 chunk 前缀 \(H_r\)。
  - Memory \(\mathcal{M}\) 来自当前策略训练轨迹；policy 模式用 **phase-local 前向窗**，operator 模式用 **global 全库**。
  - 分数：跨视角最小余弦相似作视觉支持；按臂分组归一化 L2 距离作动作风险，并做滑动平均。
  - 阈值：held-out \(\mathcal{D}_{\mathrm{cal}}\) 上按 mode 估计分位数 \(\alpha_s,\alpha_r\)（文中 pol 0.05 / op 0.30）。
  - 连续 \(L\) 次决策后切换；干预段混入下一轮（旧:新 = 2:1）。
- **对 wiki 的映射：**
  - [AutoIntervene 实体](../../wiki/entities/paper-autointervene.md) — 流程总览
  - [Why Action Chunking Improves BC](../../wiki/entities/paper-why-action-chunking-improves-bc.md)
  - [双臂操作](../../wiki/tasks/bimanual-manipulation.md)

### 3) 九任务真机（§IV / 项目页表）

- **链接：** Experiments / 项目页 Tables
- **核心贡献：**
  - ALOHA 式 leader–follower + TriPilot-FF 力反馈；ACT / DP / FM 头兼容；DINOv3 ConvNeXt-Base；\(H=100\)，评测 5 Hz（策略 30 Hz）。
  - 主七任务：Initial avg **30.9%** → AutoIntervene R2 **80.0%**（Δt 操作员时间 avg **122.9 s**）vs Human R2 **68.6%**（179.9 s）vs Additional Full Data **56.0%**。
  - 长程：Two-Towel Box Packing 28%→88%（R3）；Towels-and-Cable Bagging 8%→48%。
  - 动作头：ACT/DP/FM 均可抬升；受控 handoff 上相对 LazyDAgger / RND-DAgger 的 cut-in/out 召回与虚警显著更好。
- **对 wiki 的映射：**
  - [双臂操作](../../wiki/tasks/bimanual-manipulation.md)
  - [VLA 部署指南](../../wiki/queries/vla-deployment-guide.md)
  - [ROVE](../../wiki/entities/paper-rove-humanoid-vla-intervention.md) — 人形全身干预对照（任务层不同）

### 4) 开源边界（步骤 2.5）

- **链接：** <https://aus.bot/AutoIntervene/>、<https://aus.bot/research/autointervene/>
- **核心贡献：** 研究站与项目站均强调方法/视频/结果；**无训练代码入口**。`github.com/123qwedsa123/AutoIntervene` 描述为 Project page，树内仅静态页。截至入库日 **确认未开源（无可运行实现）**。
- **对 wiki 的映射：**
  - [项目页归档](../sites/aus-bot-autointervene.md)
  - [AutoIntervene 实体](../../wiki/entities/paper-autointervene.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-autointervene.md`](../../wiki/entities/paper-autointervene.md)
- 项目页归档：[`sources/sites/aus-bot-autointervene.md`](../sites/aus-bot-autointervene.md)
- 互链参考：[Action Chunking](../../wiki/methods/action-chunking.md)、[DAgger](../../wiki/methods/dagger.md)、[Why Action Chunking Improves BC](../../wiki/entities/paper-why-action-chunking-improves-bc.md)、[双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[VLA](../../wiki/methods/vla.md)、[ROVE](../../wiki/entities/paper-rove-humanoid-vla-intervention.md)

## BibTeX（项目页）

```bibtex
@article{autointervene2026,
  author  = {Jinhe Tang and Weiming Zhi},
  title   = {AutoIntervene: Calibrated Intervention for Action-Chunking Imitation Learning Policies},
  year    = {2026},
  eprint  = {2608.07065},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
