# LUCID: Latent-Skill Unified Control via Imagined Dynamics for Long-Horizon Humanoid Loco-Manipulation（arXiv:2608.07746）

> 来源归档（ingest）

- **标题：** LUCID: Latent-Skill Unified Control via Imagined Dynamics for Long-Horizon Humanoid Loco-Manipulation
- **缩写 / 框架：** **LUCID**（Latent-Skill Unified Control via Imagined Dynamics）
- **类型：** paper / humanoid / loco-manipulation / hierarchical-rl / model-based-rl / world-model / latent-skill
- **arXiv：** <https://arxiv.org/abs/2608.07746>（Submitted 2026-08-07；PDF：<https://arxiv.org/pdf/2608.07746>；HTML：<https://arxiv.org/html/2608.07746>）
- **DOI：** <https://doi.org/10.48550/arXiv.2608.07746>
- **作者：** Cheng Guo（通讯）、Mingzhe Ni、Angelo Cangelosi、Arash Ajoudani
- **机构：** 曼彻斯特大学计算机科学系（University of Manchester, Department of Computer Science）；意大利技术研究院人机界面与交互实验室（Istituto Italiano di Tecnologia, Human-Robot Interfaces and Interaction Lab）
- **入库日期：** 2026-08-12
- **一句话说明：** 分层 model-based RL：先用对抗模仿训结构化 latent 条件 LLC 并冻结，再联合训 macro-dynamics 世界模型与 HLC，用技能级想象 rollout 做长时程人形多物体重排，而非脚本化 FSM / 纯 model-free 技能调度。

## 开源状态（步骤 2.5）

- **项目页：** 论文与 arXiv abs **未列出** 独立项目页（`*.github.io` / lab 页）。
- **代码核查（2026-08-12）：** abs「Code, Data, Media」区无官方 GitHub；全文与 HTML 版亦无 “code will be released” / 仓库 URL。CatalyzeX 等第三方入口不算官方开源。
- **结论：** **确认未开源**（截至入库日无可运行训练/推理仓库）。wiki 局限中写明；**源码运行时序图 = 不适用**。

## 摘录 1：问题与主张（Abstract / §Introduction）

- **痛点：** 长时程人形 loco-manipulation（如多物体重排）需要可复用全身技能 **与** 可靠高层决策；现有路线常用脚本规划器、FSM、或任务特化 model-free 高层策略衔接预训练技能，**难以预测当前技能选择如何改变后续子任务条件**。
- **主张：** **LUCID** = 分层 model-based RL——在可复用技能上通过 **学得动力学模型的想象 rollout** 做规划。
- **两阶段：** (1) 对抗模仿训 **结构化 latent 条件 LLC**；(2) **冻结 LLC**，联合训 **HLC + macro-dynamics world model**。WM 预测 latent 决策诱导的 **时间扩展状态转移**（人形 / 物体 / 任务进度），而非逐步关节动力学。
- **三项贡献：** 想象动力学上的任务规划框架；交互感知课程 + 结构化 latent 接口的 LLC；多物体重排评测与消融。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-lucid.md`](../../wiki/entities/paper-lucid.md)；任务页 [`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md)；方法对照 [`wiki/methods/ase.md`](../../wiki/methods/ase.md)、[`wiki/methods/model-based-rl.md`](../../wiki/methods/model-based-rl.md)、[`wiki/concepts/latent-imagination.md`](../../wiki/concepts/latent-imagination.md)。

## 摘录 2：方法栈（§Method）

| 模块 | 要点 |
|------|------|
| **问题形式** | 目标条件 MDP；目标 \(g=(g^1,\ldots,g^M)\) 为有序物体重排子目标序列；动作 = 关节 PD 目标 |
| **LLC** | \(\pi_L(a_t\mid s_t^L,z_t)\)；奖励 \(r^D\)（对抗模仿）+ \(\lambda_G r^G\)（任务）；训练期 oracle \(\psi\) 按交互阶段供 latent |
| **结构化 latent** | 相对 ASE 无结构超球面：为 \(N\) 个语义技能保留 one-hot **anchor**，剩余维 \(\sigma_z\epsilon\) 作 within-skill 变分，再 L2 归一化（式 3） |
| **课程** | 三阶段：carry → rearrangement → retreat & chaining（含 transit、子任务完成 bonus、放置后撤退） |
| **Macro 动作** | \(a_t^H=(z_t,p_t^g)\)；\(p_t^g\) = 有界 guidance offset + waypoint-advance gate；冻结 LLC 执行最多 \(K\) 步形成一条 macro transition |
| **WM** | 紧凑状态 \(s^H=(s^c,s^f)\)（连续任务量 + 二值进度旗标）；残差预测 \(\Delta s^c\)、sigmoid 预测旗标、continuation head \(\hat\zeta\) 折扣想象回报 |
| **HLC** | DreamerV3 风格 actor–critic；从 replay 采样起点，WM 想象 \(Q\) 步；score-function + 熵；早期 \(\beta_{BC}\) 从几何 oracle 退火到 0 |
| **训练循环** | 先随机策略预填 buffer 并预训练 WM；再交替：真机（仿真）采集 → 更新 WM → 想象更新 actor/critic |

**对 wiki 的映射：** 实体页画「LLC 冻结 → WM/HLC 联合」流程图；强调 **技能级动力学** vs 关节级世界模型。

## 摘录 3：实验与消融（§Experiments）

- **仿真：** Isaac Gym；15 rigid bodies / 28 PD 关节人形（引用 scene-interaction 设定）；60 Hz 仿真、LLC 30 Hz、HLC 每 20 LLC 步一次；想象 horizon 12；8192 并行环境训 HLC/WM；LLC 两张 RTX 4090。
- **数据：** 任务自 HITR 构图；ID 62 / OOD 20；标准两物体链，扩展至五物体；LLC 参考运动含 OMOMO、SAMP。
- **指标：** SRk（前 \(k\) 物体放置成功，阈值 0.2 m）、APE、成功 episode 平均 Time（基线有 handoff reset，Time 仅描述性）。
- **基线：** InterMimic、TokenHSI（脚本 FSM）、HumanVLA（顺序激活单物体策略）；均用任务适配器，子任务交接时重置人形到有利姿态——相对 LUCID **无交接重置** 的评测更偏基线有利。
- **主结果（Table 1）：** ID SR2 **73.4%** vs 最强基线 HumanVLA **39.8%**（+33.6 pp）；OOD SR2 **68.4%** vs **37.0%**（+31.4 pp）；APE 最低。
- **长链（Fig.3）：** LUCID SR3≈56%、SR5≈21%；基线到 SR4/SR5 接近 0。
- **WM 消融（Table 2）：** 想象训练相对 model-free HLC 在稀疏/稠密奖励 × ID/OOD 上 SR2 全面更高；稠密 shaping 略抬 ID、明显伤 OOD。
- **结构化接口（Table 3）：** struct SR2 ID **74.2%** / OOD **66.4%**；unstruct **0%** SR2——可解码 ≠ 任务对齐。

**对 wiki 的映射：** 结论节写「结构化技能接口 + 宏动力学想象」双支柱；对比 TokenHSI/HumanVLA/InterMimic 与 ASE/Dreamer 谱系。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-lucid.md`**（完整论文实体骨架 + 结论；源码时序图标注不适用）。
- 更新 **`wiki/tasks/loco-manipulation.md`**：增补「技能级世界模型 + 想象 HLC」技术路线条目。
- 交叉链：[ASE](../../wiki/methods/ase.md)、[Model-Based RL](../../wiki/methods/model-based-rl.md)、[Latent Imagination](../../wiki/concepts/latent-imagination.md)、[DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[TokenHSI](../../wiki/entities/paper-bfm-38-tokenhsi.md)、[InterMimic](../../wiki/entities/paper-bfm-15-intermimic.md)。
