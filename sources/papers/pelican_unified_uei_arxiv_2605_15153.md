# Pelican-Unified 1.0：统一具身智能模型（UEI）技术报告

- **类型**：论文 / 技术报告（预印本）
- **收录日期**：2026-05-16
- **机构**：北京创新中心人形机器人（X-Humanoid）WFM System Group（作者邮箱域 `x-humanoid.com`）
- **arXiv（摘要页）**：<https://arxiv.org/abs/2605.15153>
- **PDF**：<https://arxiv.org/pdf/2605.15153.pdf>（用户给定链接与 arXiv PDF 一致）

> 版本更正：本次通读 arXiv:2605.15153v2（2026-05-21），标题为 **Pelican-Unify1.0**；沿用既有归档路径，不新建重复论文。

## 一句话

主张 **理解、推理、想象、执行** 不是四个可独立堆叠的专家模块，而应在 **共享表示 + 共同训练** 下构成单一 **统一具身智能（Unified Embodied Intelligence, UEI）** 闭环；并给出 **Pelican-Unified 1.0**：以 **Qwen3-VL-4B** 为统一理解与推理骨干，将末态隐变量 **\(z\)** 送入基于 **Wan2.2-5B** 的 **Unified Future Generator（UFG）**，在同一扩散去噪过程中 **联合生成未来视频与低层动作**，语言 / 视频 / 动作损失回传到共享表示。

## 为什么值得保留

- 把「统一」从口号落到 **三条可检验性质**：统一理解（场景、指令、视觉上下文、动作史进同一语义空间）、统一推理（语言可监督的 CoT，末态投影为 **\(z\)** 并直接约束生成）、统一生成（未来想象与动作来自 **同一条件扩散** 与同一 **\(z\)**；v2 正文以共享注意力与两路输出头描述，未给独立 action-refine 模块的实现细节）。
- 与仓库内 [WAM 综述](../../wiki/concepts/world-action-models.md) 的 **Joint WAM** 叙事高度同构，但显式引入 **VLM 推理潜变量** 作为世界–动作生成的枢纽，便于和纯 VLA、无显式推理链的 WAM 实现对照。
- 报告给出跨 **VLM 基准族**、**WorldArena**、**RoboTwin** 与 **UR5e / 天工人形** 工业面板操作等结果，可作为「单 checkpoint 多能力」主张的索引入口。

## 核心摘录（面向 wiki 编译）

### 范式层（Introduction / Abstract）

- **反对碎片化**：具身基础模型不应建立在彼此独立的 VLM、世界模型、策略头上；进步来自在可对齐、可抽象、可规划、可细化的 **潜空间世界** 中整合理解、推理、想象与执行。
- **“统一”非拼接**：不是多专家输出级联，也不是独立优化模块顺序管线；而是 **结构共享表示、相互约束条件、同一训练过程共演化**。
- **具身认知旁证**：引用神经科学 / 哲学文献支持「推理–想象–运动」在生物系统中不可分（正文引用 Kandel、Clark、Jeannerod、Friston 等，细节以 PDF 为准）。

### 架构层（§2 与 Fig.1 口径）

| 模块 | 角色（报告叙述） |
|------|------------------|
| **VLM（Qwen3-VL 4B）** | **统一理解**：场景、指令、视觉上下文、动作史 → 共享语义表示；**统一推理**：单前向自回归产生面向任务 / 动作 / 未来的 CoT，末隐状态投影为稠密 **\(z\)** |
| **UFG（Wan2.2-5B 初始化）** | **统一生成**：以 **\(z\)** 为条件，在同一 denoising 中经 **两路模态头** 联合生成 **未来视频 token** 与 **下一动作块 token**；共享注意力交换视频与动作信息，分别经模态头读出 |
| **训练** | 语言、视频、动作损失均 **反传进共享表示**（**\(z\)** 路径与 VLM 骨干），强调非「三专家后缝合」 |

### 结果数字（摘要口径，以论文表格为准）

- 八个 VLM 基准平均 **64.7**（报告称同规模可比模型中领先档）。
- **WorldArena** EWM Score **66.03**（报告称第一）。
- **RoboTwin** 平均成功率 **93.5%**（报告称在所对比动作方法中均分第二）。
- 真机：**UR5e** 与 **天工（Tienkung）人形** 在工业控制面板类任务上，展示组合与未见任务迁移；v2 未给完整真机 rollout 次数和成功率表，不能据此认定显著领先。

## 相关资料（仓库内外）

- **WAM 综述（同主题坐标）**：[sources/papers/world_action_models_survey_2605.md](./world_action_models_survey_2605.md) — 联合 \(p(o',a\mid o,l)\) 范式与 Cascaded/Joint 分类。
- **OpenMOSS Awesome-WAM**：[sources/repos/awesome-wam-openmoss.md](../repos/awesome-wam-openmoss.md) · [静态站点](../sites/awesome-wam-openmoss.md)。
- **同骨干路线的开源 VLA 参照**：[sources/papers/star_vla.md](./star_vla.md)（Qwen3-VL + 轻量动作头）；Pelican-Unified 在之上叠 **推理 \(z\) + 联合视频–动作扩散**。
- **潜空间世界–动作（对照阅读）**：[sources/papers/being_h07.md](./being_h07.md) — 另一路「世界信号进训练、像素不滚 online」的工程折中。

## 对 wiki 的映射

- 升格页面：[Pelican-Unified 1.0（UEI）](../../wiki/methods/pelican-unified-1.md) — 闭环定义、三统一、与 VLA / Joint WAM 分界及流程图。
- 交叉补强：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)。

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2605.15153v2 全文 §1–§4、所有表格、真机设置、结语与贡献者说明；无另列附录，图表按文字层核对，参考文献浏览；X-Humanoid 官方资源和 TienKung_Docs 仓库说明；未运行模型或观看视频。
- **通读：** 已完成
- Qwen3-VL-4B 自回归 CoT（Chain of Thought，不是 Cost of Transport）末隐状态投影为 z，作为 Wan2.2 UFG 的条件。视频/动作嵌入在理解和生成间共享；共享注意力后分别输出，勿把未展开的 action-refine 当成已核实独立子模块。
- 语言交叉熵、带未来 mask 的视频 MSE、带有效维 mask 的动作 Smooth L1 共同训练；正文未明确各项权重数值，不补造配置。
- VLM 平均 64.7 将部分基线不支持项按零计，不能当作同能力覆盖的公平均值。RoboTwin clean/randomized 为 93.6/93.3、平均 93.5，低于该表 Motubrain 95.9。
- WorldArena EWM 66.03 属该比较集合；人工评价均值 1.76 来自四个 0/1/2 分轴，不是成功率。
- UR5e 组合任务只训练原子技能；天工五个已见任务各约 300 条视频–动作、三个未见任务各 50 条视频。后者为无对应机器人动作监督的迁移，不是完全未接触任务视频。缺完整真机样本数/成功率表。
- 组织或硬件 TienKung_Docs 仓库不构成本模型训练/推理代码。未核实到 Pelican-Unify1.0 专属可运行实现、权重与处理后训练数据；无源码调用图可供确认。
