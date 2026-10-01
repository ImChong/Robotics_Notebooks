# LocoWM: High-Precision Locomotion through World-Model-Guided Residual Adaptation（arXiv:2609.39179）

> 来源归档（ingest）

- **标题：** LocoWM: High-Precision Locomotion through World-Model-Guided Residual Adaptation
- **简称：** LocoWM
- **类型：** paper / wheeled-legged / world-model / residual-rl / payload-transport
- **arXiv：** <https://arxiv.org/abs/2609.39179>
- **PDF：** <https://arxiv.org/pdf/2609.39179>
- **项目页：** <https://zhaozijie2022.github.io/LocoWM> — 归档见 [`sources/sites/locowm.md`](../sites/locowm.md)
- **代码：** <https://github.com/zhaozijie2022/LocoWM> — 归档见 [`sources/repos/locowm.md`](../repos/locowm.md)
- **机构：** 中国科学院自动化研究所（CASIA）、中国科学院大学（UCAS）、北京邮电大学（BUPT）、北京交通大学（BJTU）
- **入库日期：** 2026-10-01
- **最后更新：** 2026-10-01
- **一句话说明：** 世界模型一次前向预测 action-conditioned 任务子状态序列，残差适配器据此做 **preactive** 修正 \(a_t=a_t^b+a_t^r\)；两阶段训练解耦行走与精度；Go2-W 真机零样本三类高精度任务，附录 G1 托盘仿真 94.1% 成功率。

## 开源状态（步骤 2.5，2026-10-01 核查）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（Paper / Code / 真机与仿真视频） |
| GitHub | **已开源** — `zhaozijie2022/LocoWM`（Isaac Lab 5.1 + RSL-RL 两阶段训练与评测脚本） |
| 预训练权重 | README 未列公开 checkpoint；需按文档自训 Stage1/2 |

**结论：已开源**（训练/评测管线完整；权重需本地训练）。

## 核心摘录

### 摘录 1：任务子状态接口

- 精度目标、世界模型预测目标、残差输入统一为 **task substate** \(z_t=h(s_t)\)（Go2-W 背载平台：pitch/roll、线加速度、角速度等）。
- 世界模型 \(f_\psi(o_t,a_t^b)\mapsto \hat{z}_{t+1:t+H}\) **单次前向**，不做在线轨迹优化。

**对 wiki 的映射：** [paper-locowm](../../wiki/entities/paper-locowm.md)

### 摘录 2：两阶段训练

- **Stage 1：** PPO 训 base policy（仅 \(r^{\mathrm{track}}\)）+ 同 rollout 上 MSE 训 action-conditioned world model。
- **Stage 2：** 冻结 base 与 WM，PPO 训 residual adapter（\(r^{\mathrm{track}}+r^{\mathrm{prec}}\)）。

**对 wiki 的映射：** [paper-locowm](../../wiki/entities/paper-locowm.md)

### 摘录 3：评测与真机

- 仿真：**terrain leveling**（slope / bump / bridge / rough）、**acceleration compensation**、**push recovery**；对照 Base / End-to-End / React / Recon。
- 相对 Recon，成功率最高 **+29.6 pp**；roll RMS 最多降 **91.8%**，峰值竖直加速度最多降 **83.0%**。
- **Unitree Go2-W** 真机零样本：非固定载荷越障、加减速倾角补偿、推扰恢复。
- 附录 **Unitree G1** 双手托盘仿真：**94.1%** vs ReST-RL **91.0%**。

**对 wiki 的映射：** [paper-locowm](../../wiki/entities/paper-locowm.md)、[wheel-legged-quadruped](../../wiki/concepts/wheel-legged-quadruped.md)

## 当前提炼状态

- [x] 项目页与 GitHub 核查
- [x] wiki 实体页
- [x] 交叉更新轮足概念页与 loco-manipulation 任务页
