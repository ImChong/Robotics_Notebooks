# NestDex（arXiv:2608.13362）

> 来源归档（ingest）

- **标题：** NestDex: Nested Policy Learning with Copilot Assisted Teleoperation for Dexterous Manipulation
- **缩写：** **NestDex**
- **类型：** paper / dexterous-manipulation / teleoperation / shared-autonomy / imitation-learning / action-chunking
- **arXiv：** <https://arxiv.org/abs/2608.13362>
- **HTML：** <https://arxiv.org/html/2608.13362>
- **PDF：** <https://arxiv.org/pdf/2608.13362>
- **提交：** 2026-08-13（arXiv v1）
- **项目页：** <https://aus.bot/research/nestdex/> — 归档见 [`sources/sites/aus-bot-nestdex.md`](../sites/aus-bot-nestdex.md)
- **代码：** 截至 **2026-08-17** 项目页 **未列训练/推理仓**；GitHub 无官方实现 → **确认未开源**
- **作者：** James Zhao\*、Jinhe Tang\*、Mingyuan Ba、Weiming Zhi†（\* 共同一作，† 通讯 Weiming.Zhi@sydney.edu.au）
- **机构：** 悉尼大学计算机学院 + Australian Centre for Robotics（PAIR Lab，aus.bot）；范德堡大学 College of Connected Computing
- **入库日期：** 2026-08-17
- **一句话说明：** 嵌套策略：可复用本体感觉内层手技能 + 单自由度 clutch copilot 采完整任务示范，再训**部署时不再依赖内层**的外层 visuomotor；H-VAE 压缩 20-DoF 手指令。六任务真机相对 AnyTeleop 采数成功率与外层策略成功率均更高。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2608.13362>
- **核心贡献：** 灵巧示范难在「臂往哪走」与「手指怎么接触」必须同时指定。NestDex 把学到的手技能放进采数环：操作员控臂，用 1-DoF clutch 调节当前内层手策略的进度；VLM 按任务阶段选技能。内层**只协助采数**，不进入最终控制器。完整示范再训外层 visuomotor（臂关节 + 手 latent）。
- **对 wiki 的映射：**
  - [NestDex 实体](../../wiki/entities/paper-nestdex.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)

### 2) 内层技能、clutch 与 VLM 选择（§IV-A / §IV-B）

- **链接：** Method
- **核心贡献：**
  - 多视三角化人手关键点；AnyTeleop 向量重定向改 Huber 残差 + 时序平滑，得到 follower 手关节与力矩轨迹。
  - 每技能一条本体感觉 action-chunk 策略：历史 \(h=30\) 的 \((\mathbf{q},\mathbf{e})\) → \(H_{\mathrm{in}}=30\) 关节块；grasp 技能跨四物体共训，部署**无图像/物体 ID**。
  - clutch 把标量进度 \(p_t\in[0,1]\) 映射到执行索引；每周期最多 ±1 步，可正向、反向、保持；反向清 ensemble 但保留轨迹缓冲。
  - VLM 读腕相机 + 技能描述，仅在启动或完全退回 \(r^{\mathrm{in}}=0\) 时切换技能。
- **对 wiki 的映射：**
  - [NestDex 实体](../../wiki/entities/paper-nestdex.md) — 流程总览
  - [数据手套 vs 视觉遥操作](../../wiki/comparisons/data-gloves-vs-vision-teleop.md)
  - [深度遥操作路线 Stage 4](../../roadmap/depth-teleoperation.md)

### 3) 外层 BC、H-VAE 与时间集成（§III / §IV-C）

- **链接：** Preliminaries + Outer Policy
- **核心贡献：**
  - 内外层均用 ACT 式 action chunk + 指数时间集成 \(w_i=\exp(-mi)\)。
  - H-VAE 把 20-DoF 手指令编到 10-D posterior mean 作 BC 目标；臂指令保持关节空间；部署时 decoder 还原手关节。
  - 外层：DINOv3 视觉 + Transformer，\(H=100\)，腕相机 \(256\times256\)。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md)
  - [AutoIntervene](../../wiki/entities/paper-autointervene.md) — 同实验室：部署期 chunk 接管 vs 本文采数期 copilot

### 4) 六任务真机（§V / Table I–III）

- **链接：** Empirical Evaluation
- **核心贡献：**
  - 平台：leader 7-DoF Piper Nero + 1-DoF clutch；follower 同臂 + **WujiHand I（20-DoF）** + 腕相机；内层 100 Hz。
  - 采数（每法 20 次，Table II）：Copilot 六任务 **100%**；AnyTeleop 在 Tongs / Toast / Binder 为 **0%**，其余 30–75%，且成功示范更慢。
  - 外层（20 rollout，Table III）：Copilot+H-VAE 四任务 **100 / 75 / 90 / 100%**，全面优于无 H-VAE 与 AnyTeleop 示范。
  - 接触消融（瓶抓）：开环回放 3/10；闭环无 ensemble 7/10；闭环+时间集成 **9/10**（vs 回放 \(p=0.0198\)）；无 ensemble 的 P95 jerk 约 **2.30×**。
- **对 wiki 的映射：**
  - [双臂操作](../../wiki/tasks/bimanual-manipulation.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [TeleDexter](../../wiki/entities/paper-teledexter.md) — 灵巧遥操作另一条「低层执行」路线

### 5) 开源边界（步骤 2.5）

- **链接：** <https://aus.bot/research/nestdex/>
- **核心贡献：** 研究站强调方法 / 视频 / 论文；**无训练代码入口**。截至入库日 **确认未开源（无可运行实现）**。
- **对 wiki 的映射：**
  - [项目页归档](../sites/aus-bot-nestdex.md)
  - [NestDex 实体](../../wiki/entities/paper-nestdex.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-nestdex.md`](../../wiki/entities/paper-nestdex.md)
- 项目页归档：[`sources/sites/aus-bot-nestdex.md`](../sites/aus-bot-nestdex.md)
- 互链参考：[Teleoperation](../../wiki/tasks/teleoperation.md)、[灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Behavior Cloning](../../wiki/methods/behavior-cloning.md)、[AutoIntervene](../../wiki/entities/paper-autointervene.md)、[TeleDexter](../../wiki/entities/paper-teledexter.md)、[双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[深度遥操作路线](../../roadmap/depth-teleoperation.md)

## BibTeX（项目页）

```bibtex
@misc{zhao2026nestdexnestedpolicylearning,
  title         = {NestDex: Nested Policy Learning with Copilot Assisted Teleoperation for Dexterous Manipulation},
  author        = {James Zhao and Jinhe Tang and Mingyuan Ba and Weiming Zhi},
  year          = {2026},
  eprint        = {2608.13362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.13362}
}
```
