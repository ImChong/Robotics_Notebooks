# HumoSlope: Physics-Guided Biomechanical Gait Adaptation for Humanoid Locomotion on Extreme Sloped Terrains（arXiv:2607.07830）

> 来源归档（ingest）

- **标题：** Physics-Guided Biomechanical Gait Adaptation for Humanoid Locomotion on Extreme Sloped Terrains
- **类型：** paper / humanoid locomotion / sloped terrain / physics-guided RL / biomechanical prior / sim2real
- **框架短名：** HumoSlope
- **arXiv abs：** <https://arxiv.org/abs/2607.07830>
- **arXiv HTML：** <https://arxiv.org/html/2607.07830v1>
- **PDF：** <https://arxiv.org/pdf/2607.07830>
- **项目页 / 代码：** 截至入库日（2026-07-31）arXiv 摘要与 HTML 全文**未列**项目页或 GitHub；论文亦未写 “code will be released”。按**确认未开源**处理。
- **机构：** 南洋理工大学（Nanyang Technological University）、新加坡科技研究局（A*STAR）
- **作者：** Xuanyu Chen⋆、Mohan Liu⋆、Dengchen Mei、Zhihao Gu、Haitian Zhang、Kaimin Mao（NTU）；Haiyue Zhu、Shijun Yan（A*STAR）；Lin Wang†（NTU）（⋆共同一作，†通讯）
- **硬件：** Unitree G1（仿真 + 真机户外）
- **仿真 / 训练：** Isaac Lab；PPO；非对称 actor–critic；单卡 NVIDIA RTX 5090 上千并行环境；域随机化
- **发表日期：** 2026-07-08（arXiv v1）
- **入库日期：** 2026-07-31
- **一句话说明：** 两阶段物理引导框架 **HumoSlope**：Stage I 用**坡面局部支撑平面上的 slope-adaptive ZMP 正则**学盲行走平衡先验；Stage II 用训练期 PCA 地形描述子门控 **BSGA** 生物力学软先验，抑制低 CoM「Groucho」蹲姿退化；部署 actor **纯本体感知**，真机户外草地坡连续穿越至 **62.7%（32.1°）**。

## 摘要级要点

- **问题：** 陡坡对全身施加**持续重力偏置**，不同于楼梯/踏石的短时落脚选择；通用奖励下 RL 易收敛到慢速、低 CoM 蹲姿（Groucho gait），牺牲姿态质量并封顶更大坡度能力。
- **Stage I — slope-adaptive ZMP：** 在支撑脚估计的**局部倾斜支撑平面**上评估地形对齐 ZMP 偏差 \(d_{\mathrm{zmp}}^{\mathrm{ta}}\)（相对接触力加权支撑锚点 \(\mathbf{p}_{\mathrm{sa}}\)），而非世界水平面参考；用点质量表观力 \(\mathbf{F}_{\mathrm{app}}=\mathbf{g}-\mathbf{a}_{\mathrm{com}}\) 求交，奖励 \(r_{\mathrm{zmp}}^{\mathrm{ta}}=\exp(-d/\sigma)\)。仅 Stage I 使用，产出可 warm-start 的盲 locomotion 先验。
- **Stage II — BSGA：** 对高度扫描点做 PCA，得到 5 维宏观描述子 \(\boldsymbol{\phi}^{\mathrm{PCA}}_{5}=(\theta_{\mathrm{slope}},\theta_{\mathrm{bank}},|\theta_{\mathrm{slope}}|,\mathbbm{1}_{\mathrm{up}},\mathbbm{1}_{\mathrm{down}})\)（训练特权）；门控 \(r_{\mathrm{BSGA}}^{\mathrm{core}}=w_{\mathrm{com}}r_{\mathrm{com}}+w_{\mathrm{bio}}r_{\mathrm{bio}}+w_{\mathrm{swing}}r_{\mathrm{swing}}\)：坡条件 CoM 高度、上坡髋主导推进、下坡膝制动/控步，以及由 Stage I rollout 标定的摆动髋俯仰软跟踪。
- **部署：** actor 观测仅含角速度、投影重力、速度命令、关节状态与动作历史；**无在线外感知**。
- **仿真主结果（held-out 复合坡道，摩擦三档平均）：** 相对 URL / FastTD3 / Gallant（深度），Ours 在 \(0^\circ\)–\(20^\circ\) 近满分且穿越更快；\(30^\circ\) 仍有 **77.1% SR**（基线 0%），最大坡度 sweep 达 **73%（36°）**。
- **消融（20°）：** 去掉 slope-adaptive ZMP → SR 55.6%；去掉 BSGA 奖励先验 → 26.9%；去掉整块 BSGA → 0%；Stage I only 虽 100% SR 但 CoM 更低、更慢（保守蹲姿）。
- **真机：** 沥青/波浪/路边/两处草地坡 + 平地走道；最陡草地均值 **32.1°**（局部至 36.4°）；雨后湿滑亦可；姿态随上下坡前倾/直立/微后仰变化。
- **局限（作者自述）：** 盲策略无法预见坡变，姿态调整在接触后才出现；突变障碍、可变形地面、极不规则户外仍受限；未来可加可选视觉 look-ahead。

## 核心摘录（面向 wiki 编译）

### 两阶段与关键奖励

| 阶段 | 作用 | 关键机制 |
|------|------|----------|
| Stage I | 地形一致平衡先验 | slope-adaptive ZMP 正则（局部支撑平面） |
| Stage II | 抑制蹲姿退化 + 上下坡非对称步态 | PCA 描述子门控 BSGA（CoM / bio / swing） |
| Deploy | 盲部署 | 纯本体感知 actor；特权描述子仅训练期 |

### 与基线对比（索引级）

| 维度 | HumoSlope | Unitree RL Lab | FastTD3 | Gallant |
|------|-----------|----------------|---------|---------|
| 输入（部署） | **本体感知** | 本体感知 | 本体感知 | **深度外感知** |
| 坡专用物理先验 | **局部平面 ZMP** | 通用奖励 | 通用 | 体素/感知 |
| 姿势控制 | **BSGA 坡条件软先验** | 易低 CoM 蹲姿 | — | 感知辅助 |
| 仿真 30° SR | **77.1%** | 0% | 0% | 0% |
| 真机最大连续坡 | **草地 32.1°** | — | — | — |

## 对 wiki 的映射

- 沉淀实体页：[HumoSlope 极端坡面生物力学步态适应（arXiv:2607.07830）](../../wiki/entities/paper-humoslope-physics-guided-slope-locomotion.md)
- 交叉补强：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[LIP / ZMP](../../wiki/concepts/lip-zmp.md)、[Privileged Training](../../wiki/concepts/privileged-training.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[unitree_rl_lab](../../wiki/entities/unitree-rl-lab.md)、[Isaac Lab](../../wiki/entities/isaac-lab.md)、[PPO](../../wiki/methods/ppo.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 摘要、两阶段架构、ZMP/BSGA 机制、仿真与真机要点摘录
- [x] wiki 实体页与 locomotion / ZMP / terrain 交叉链接规划
- [x] 项目页/代码核查：截至 2026-07-31 **确认未开源**（无 URL、无将开源声明）
- [ ] 若作者后续公开代码/权重，补 `sources/repos/` 与源码运行时序图
