# GaitSpan: Growing Humanoid Locomotion from Walking to Running（arXiv:2607.12114）

> 来源归档（ingest）

- **标题：** GaitSpan: Growing Humanoid Locomotion from Walking to Running
- **类型：** paper / humanoid locomotion / skill growth / gait emergence / continuous speed / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2607.12114>
- **arXiv HTML：** <https://arxiv.org/html/2607.12114v1>
- **PDF：** <https://arxiv.org/pdf/2607.12114>
- **项目页：** <https://gaitspan2026.github.io/>
- **机构：** 密歇根大学（University of Michigan）、加州大学伯克利分校（UC Berkeley）、美国 Skyline High School
- **作者：** Kwan-Yee Lin†、Zilin Wang†（UMich，共同一作）、Janelle J. Liu（Skyline High School）、Stella X. Yu（UMich / UC Berkeley）
- **硬件：** Booster T1、Booster K1、Unitree G1（真机）；五套人形 embodiment（23/29 DoF 等）仿真与 sim-to-sim
- **仿真：** NVIDIA Isaac Gym（200 Hz 物理 / 50 Hz 控制，4096 并行环境）；训练 FastSAC；sim-to-sim 评估 MuJoCo
- **种子策略：** 预训练 **行走策略** $\pi_{\mathrm{seed}}$（冻结；观测含正弦相位编码，与 [Holosoma](../../wiki/entities/holosoma.md) 类 locomotion 管线一致）；训练期将平面速度命令映射到种子策略的 $[-1,1]$ 归一化域
- **发表日期：** 2026-07-13
- **入库日期：** 2026-07-16
- **一句话说明：** 把 **冻结行走策略** 当作 **种子技能**，经 **GaitWave**（多相位尺度重查询 + 分层速度记忆组合 action waves）、**H-SLIP**（三层虚拟腿 SLIP 动态塑形）与 **残差分支**，在 **无步态标签、无人体演示、无多专家蒸馏** 前提下，用 **单一速度条件策略** 连续覆盖走–慢跑–跑类 regime，并在 **五套 embodiment** 与 **未见真机地形** 上零样本部署。

## 摘要级要点

- **问题：** 现有 diverse-speed 人形 locomotion 多靠 **步态日程表、动作片段模仿、分专家训练再蒸馏/切换**；对 **连续速度命令、地形与形态** 的灵活性有限，且 regime 边界易 **平均化、切换脆弱**。
- **核心范式 — skill growth：** 行走不是终点，而是可再生的 **运动结构先验**（平衡、支撑、全身协调、接触切换）；更快 locomotion 应 **再生节律、拉长步幅、残差修正**，而非从零重学。
- **GaitWave：** 对冻结 $\pi_{\mathrm{seed}}$ 用相位尺度 $\mathcal{P}=\{\rho_0,\rho_2,\ldots,\rho_K\}$ 生成 **canonical action waves** $\mathbf{a}^{\mathrm{seed}^{(k)}}_t$；**分层多分辨率记忆库**（2/4/8 bin + SoftBlend）学习速度条件组合系数 $\boldsymbol{\alpha}_t$，避免硬分区步态切换。
- **H-SLIP：** 三层虚拟腿（root–foot / root–knee / knee–foot）塑形 **压缩、回弹、触地、腾空** 四阶段；奖励含速度跟踪门控 $g^{\mathrm{track}}$、触地足端速度惩罚与 **flight** 项，引导高速进入慢跑/跑类接触模式。
- **总动作：** $\mathbf{a}_t = \mathbf{a}^{\mathrm{wave}}_t + \mathbf{a}^{\mathrm{res}}_t$（clip）；$\mathbf{a}^{\mathrm{wave}}$ 由 GaitWave 组合种子波，$\mathbf{a}^{\mathrm{res}}$ 为可训练残差策略。
- **自锚正则 $\mathcal{L}_{\mathrm{SA}}$：** 低速种子锚定、残差紧凑、相位自相似、相对种子的时间一致性——防止扩张时覆写行走稳定性。
- **训练：** FastSAC；命令速度 **2/4/8 分辨率并行**；高速采样偏置在训练前 95% 从 1.0 线性增至 2.6，质量集中于 **~2.2 m/s**；Booster T1/K1 **100K** iter，G1 **50K** iter。
- **评测指标：** 速度跟踪误差、**Flight Time**（无接触腾空时长）、**Energy**（关节力矩–速度积分）。
- **真机展示（项目页）：** 波场/砂石路走（~0.5–0.7 m/s）、走→慢跑（0.5→1.1 m/s）、草地跑（~2.8 m/s）、砾石跑（~2.5 m/s）、跑→走（2.4→0.5 m/s）、林间坡道慢跑长程（~1.1 m/s）；负重 + 低摩擦鞋扰动下 0.5→2.5 m/s 加速；**IsaacGym→MuJoCo** 五 embodiment 零样本 sim-to-sim。

## 核心摘录（面向 wiki 编译）

### 三模块与 vanilla 种子–残差对比

| 模块 | 作用 | 单独使用局限（论文消融） |
|------|------|--------------------------|
| **Vanilla seed–residual** | $\mathbf{a}=\mathrm{clip}(\mathbf{a}^{\mathrm{seed}}+\mathbf{a}^{\mathrm{res}})$ | 高速跟踪差，难以产生持续 flight |
| **GaitWave** | 相位尺度 waves + 分层记忆组合 | 低速跟踪好，但高速 flight 有限、极端 OOD 表达不足 |
| **H-SLIP** | 虚拟腿动态事件奖励 | 高速 flight 多，但 OOD 跟踪急剧恶化 |
| **GaitSpan（完整）** | 三者 + SA 正则 | 全速域跟踪与渐进 flight 的最佳折中 |

### 与相邻路线的对比（索引级）

| 维度 | GaitSpan | SD-AMP / 多专家蒸馏 | AMP / 人体演示 | SPRINT 频谱先验 |
|------|----------|---------------------|----------------|-----------------|
| 先验来源 | **机器人自身冻结行走策略** | LAFAN1 + 门控/分专家 | 人体 MoCap 对抗先验 | 5 条 LAFAN1 频谱 VAE |
| 步态标签 | **无** | walk/run 参考分区 | 隐式模仿分布 | 速度–频率锚点 |
| 连续变速 | **单策略，~0–2.5+ m/s** | 速度条件 + 双判别器 | 速度变化有限、低速也 flight | **0–6 m/s**（G1 冲刺向） |
| 形态泛化 | **五 embodiment 同一管线** | 主要 G1 | 主要 G1 | 跨身高人形 |
| 真机地形 | **训练平地/缓坡，零样本户外** | 硬件走跑起身 | 硬件走跑 | 户外冲刺验证 |
| 能耗 | 相对 AMP **中低速更省** | — | 低速也偏高能耗 | — |

### 基线设置备忘

- **Seed：** 冻结行走策略直接外推扩展命令域。
- **Energy Multi-Experts [Fu et al., CoRL 2021]：** 分速域独立能耗塑形专家再组合；高速 tracking 差、几乎无 flight。
- **Human Demonstration：** AMP [Peng et al.]；人形姿态更「像人」，但 **速度条件步态分化弱**、低速 flight/能耗偏高。
- **对比命令点：** 定性对比常用 $0.5 / 1.5 / 2.5$ m/s；种子策略训练域约 $[-1,1]$ m/s。

## 对 wiki 的映射

- 沉淀实体页：[GaitSpan 从行走到跑步的技能生长（arXiv:2607.12114）](../../wiki/entities/paper-gaitspan-humanoid-locomotion-walking-running.md)
- 交叉补强：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[SLIP + VMC](../../wiki/methods/slip-vmc.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[holosoma](../../wiki/entities/holosoma.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[SD-AMP 统一走跑起身](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[SPRINT 人形冲刺频谱先验](../../wiki/entities/paper-sprint-humanoid-athletic-sprints.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 摘要、GaitWave / H-SLIP / 残差架构、训练与基线要点摘录
- [x] wiki 实体页与 humanoid-locomotion 交叉链接规划
- [ ] 待作者公开代码/权重后补 `sources/repos/`
