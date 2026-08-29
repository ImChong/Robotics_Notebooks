# Riemann-1.0

> 来源归档（ingest）

- **标题：** Riemann-1.0: An Embodied World Action Model for Physical AI
- **缩写：** **Riemann-1.0**
- **类型：** paper / world-action-model / manipulation / company-technical-report
- **项目页：** <https://riemann-dynamics.github.io/Riemann-1.0-Website>
- **PDF：** <https://riemann-dynamics.github.io/Riemann-1.0-Website/paper/Riemann-1.0.pdf>
- **arXiv：** **无**（截至 2026-08-29）
- **代码：** **未开源**
- **作者：** Riemann Dynamics（集体署名；`research@riemanndynamics.ai`）
- **机构：** 黎曼动力（Riemann Dynamics）；昆仑万维（Kunlun Wanwei）子公司
- **入库日期：** 2026-08-29
- **一句话说明：** **全因果自回归** World Action Model：先预测动作再条件化未来视觉 latent，同一模型兼任可执行策略与动作条件世界仿真；三阶段渐进预训练吃 **232K+ h** 人视频 / UMI / 机器人轨迹；RoboCasa365 **62.6%**、天机 Marvin 真机均 **85.0% SR**。

## 核心论文摘录（MVP）

### 1) 问题：异构具身经验 + 现有 WAM 因果不统一（§1）

- **链接：** [项目页 PDF](https://riemann-dynamics.github.io/Riemann-1.0-Website/paper/Riemann-1.0.pdf)
- **摘录要点：** 具身经验跨 **egocentric 人视频、UMI/手持夹爪、异构机器人轨迹**，观测/动作空间/监督强度不一致。现有 WAM 被归为三类——**联合去噪**（DreamZero）、**视频优先再反推动作**（LingBot-VA）、**视频/动作分塔**（Fast-WAM）——都无法在 **统一因果过程** 里同时支持高效策略与动作条件仿真。Riemann-1.0 主张把交互写成 **观测–状态–动作** 的因果状态转移。
- **对 wiki 的映射：**
  - [Riemann-1.0 论文实体](../../wiki/entities/paper-riemann-1.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 数据基建：232K+ h 与六段数据引擎（§2）

- **链接：** PDF §2 / Figure 2
- **摘录要点：** 语料约 **200K+ h** 人第一人称（约 86%）、**12K+ h** UMI/外骨骼（约 5%）、**20K+ h** 机器人轨迹（约 9%）。人视频走 **VLM 分层切段 + MANO 3D 手 + VGGT-Ω 相机位姿**；UMI/机器人侧重对齐、夹爪边界与静止/抖动过滤。监督被写成连续谱：冻结 **LAM 伪动作** → 3D 手 / UMI / 机轨迹真动作 → 高质量机器人-only。按语义 taxonomy 采样，而不是按原始小时数均匀抽。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [EgoScale](../../wiki/methods/egoscale.md)（人视频缩放对照）

### 3) 全因果分解与共享 DiT（§3，式 1–4）

- **链接：** PDF §3.2 / Figure 3–4
- **摘录要点：**
  \[
  p(a_{1:T},z_{1:T}\mid z_0,s_0,c)=\prod_t p(a_t\mid z_{<t},s_{<t},a_{<t},c)\,p(z_t\mid z_{<t},s_{<t},a_{\le t},c)
  \]
  动作先于对应视觉后果；状态 **不生成**，只从环境/回放注入。骨干为 **Wan VAE + 共享 Action/Video DiT**，T5 编任务/本体/视角提示；**本体 ID** 选动作/状态投影与头。训练用结构化因果 mask + **flow matching** 双头：\(\mathcal{L}=(1-\lambda)\mathcal{L}_z+\lambda\mathcal{L}_a\)。相对 GIGA-World-Policy，作者强调 **从预训练起就用同一因果目标**，而不是先吃通用视频生成再适配策略。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint 族 **动作优先因果** 实例
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 4) 三阶段渐进预训练 + 后训练（§4）

- **链接：** PDF §4 / Figure 7
- **摘录要点：** Stage I **LAM-Action Bootstrap**（λ=0.1）：冻结 LAM（VIPRA 系 32 维潜动作 VAE）给人视频打伪动作，主学视觉动力学。Stage II **Trajectory-Grounded Alignment**（λ=0.5）：3D 手 + UMI + 机器人混合对齐。Stage III **Robot-Policy Enhancement**（λ=0.9）：仅高质量机器人轨迹。后训练把四任务遥操作数据合成一个 generalist，λ 提到 **0.95**。§4.2 写「每任务 **3 小时**」，§5.1.1 写「每任务 **15 条**」——入库时按原文并存，勿合成单一数字。
- **对 wiki 的映射：**
  - [Riemann-1.0 论文实体](../../wiki/entities/paper-riemann-1.md)
  - [Dyna-2](../../wiki/entities/dyna-2.md) — 同属人视频预训练、闭源产业 WAM

### 5) 实验要点（§5）

- **真机（天机 Marvin 双臂，Table 1）：** 均 SR **85.00%** / PSR **94.43%**；厨房整理 **90.0 / 98.4**（次强 G0.5 SR 仅 **35.0**）；积木 **85.0**、叠衣 **85.0**、桌面 **80.0**。对照含 DreamZero*（作者用官方开源码 + **自有大规模数据** 复现，非原论文数字）、τ₀-WM、LingBot-VLA / VA、π₀.₅、G0.5。
- **Held-out（Table 2，每任务 10 trial）：** 组合泛化均 **65.0%**；OOD 均 **85.0%**；总体 **75.0%**。
- **RoboCasa365 Target-50（Table 3）：** 均 **62.6%**（Atomic-Seen 74.2 / Composite-Seen 56.0 / Unseen 56.3）vs ABot-M0.5 **54.2%**（+8.4）。
- **RoboTwin 2.0（Table 4）：** Clean **94.6** / Randomized **94.0** / 均 **94.3**，与 ABot-M0.5 **94.1** 同档饱和。
- **LIBERO（Table 5）：** 均 **99.0**（Spatial 99.6 / Object 100.0 / Goal 97.6 / Long 98.6）。
- **对 wiki 的映射：**
  - [ABot-M0.5](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md)
  - [G0.5](../../wiki/entities/paper-galaxea-g05.md)
  - [τ₀-WM](../../wiki/entities/tau0-world-model.md)

### 6) 谱系、开源与局限（§6–7 / 项目页）

- **谱系：** 作者把自身写成 **动作优先全因果 AR**，对照 DreamZero 联合去噪、LingBot-VA 视频优先、Fast-WAM 分塔、GIGA-World-Policy「训时未来 / 推理动作中心」。
- **开源：** 项目页与 GitHub 组织 **未列** 模型代码或权重（2026-08-29）。
- **正文缺口：** 无独立 Limitations 节；**无消融表**（人视频小时数、λ、因果顺序等均无定量拆解）。真机 n 小（held-out 每任务 10）；LIBERO / RoboTwin 已近饱和，主增量在 RoboCasa365 与厨房长程。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [WAM 纵深路线](../../roadmap/depth-wam.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-riemann-1.md`](../../wiki/entities/paper-riemann-1.md)
- 项目页：[`sources/sites/riemann-1-0-website.md`](../sites/riemann-1-0-website.md)
- 官网仓：[`sources/repos/riemann-1-0-website.md`](../repos/riemann-1-0-website.md)
