# SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints（arXiv:2605.28549）

> 来源归档（ingest）

- **标题：** SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints
- **类型：** paper / humanoid locomotion / spectral prior / sim2real / athletic sprint
- **arXiv abs：** <https://arxiv.org/abs/2605.28549>
- **arXiv HTML：** <https://arxiv.org/html/2605.28549v1>
- **PDF：** <https://arxiv.org/pdf/2605.28549>
- **项目页（匿名）：** <https://anonymous.4open.science/w/SPRINT-138A/>
- **机构：** 国防科技大学智能科学技术学院（Yantong Wei†, Kaihong Huang†, Hainan Pan, Jiawei Luo, Jiawei Zhou, Ziyan Mai, Zhiwen Zeng, Huimin Lu*）；湖南大学人工智能与机器人学院（Yaonan Wang）
- **硬件：** Unitree G1（1.3 m 级人形，真机 field experiments）；跨形态实验另含 1.1 m / 1.7 m 人形
- **仿真：** NVIDIA Isaac Gym（200 Hz 物理、50 Hz 控制；单卡 RTX 4090 约 6.5 h 收敛）
- **入库日期：** 2026-06-05
- **一句话说明：** 用 **5 条 LAFAN1 重定向单周期步态（428 帧 / 14 s）** 训练 **频率自适应频谱先验（VAE + 多谐波 + FiLM 解码）**，再以 **冻结先验 + PPO 残差** 在 G1 上实现 **0–6 m/s 连续变速与无缝步态切换**，**零样本 sim2real** 峰值冲刺 **6 m/s**。

## 摘要级要点

- **问题：** 人形竞技冲刺缺乏可用运动学参考；纯 RL 奖励易得不自然步态；AMP 在高动态冲刺上易不稳定 / mode collapse；模仿跟踪与 AI-CPG 等难以连续外推至参考分布之外的极速域（AI-CPG 仿真峰值 < 4 m/s）。
- **核心思路：** 在频域刻画步态周期性——用极少离散参考建立 **速度–主频映射**，训练 **frequency-adaptive spectral priors** 生成 10-DoF 关节子集轨迹，再与 **低层残差 RL + PD** 分层组合。
- **参考库（Stage I）：** LAFAN1 → **GMR** 重定向 → 选 10 关节（髋/膝/踝 pitch + 肩 pitch + 肘）→ 5 条单周期步态各标准化 10 s、30→60 Hz 重采样 + Savitzky–Golay → FFT 得 PSD 与对侧半周期相位关系。
- **速度–频率锚点：** $\{0.66,1.10,2.29,2.87,3.40\}$ m/s ↔ $\{0.68,0.86,1.25,1.36,1.58\}$ Hz（走/慢跑/跑）。
- **频谱先验（Stage II）：** 频率条件 VAE 潜变量 + $K{=}6$ 多谐波相位向量 + FiLM 调制 MLP 解码；损失 $L{=}L_{\mathrm{rec}}+\beta_{\mathrm{KL}}L_{\mathrm{KL}}$。
- **SPRINT 策略（Stage III）：** 冻结先验输出 $q^{\mathrm{ref}}$，PPO 输出残差 $a^{\mathrm{res}}$，$q^{\mathrm{target}}=q^{\mathrm{ref}}+\alpha a^{\mathrm{res}}$；奖励含任务跟踪、先验贴合 $r_{\mathrm{prior}}$、足端引导 $r_{\mathrm{feet}}$、正则 $r_{\mathrm{reg}}$；**非对称 Actor–Critic**（critic 含线速度与先验参考）、渐进速度课程、动力学随机化 → **零样本真机**。
- **主要结果（相对 Humanoid-Gym / AMP / AI-CPG）：** 先验侧 $L_{\mathrm{rec}}$、FID、$E_{\mathrm{BA}}$ 优于 AI-CPG；策略侧峰值 **6 m/s**、更低 FID 与 $E_{\mathrm{qpos}},E_{\mathrm{vel}}$、更快收敛；真机与仿真指标在全速域一致。

## 核心摘录（面向 wiki 编译）

### 三阶段管线

| 阶段 | 输入 | 输出 |
|------|------|------|
| I 参考库 | LAFAN1 人类步态 + GMR | 5×10-DoF 单周期轨迹 + 速度–频率表 |
| II 频谱先验 | 目标频率 $f$、时间 $t$ | 关节参考轨迹 $q^{\mathrm{ref}}(t,f)$，可 OOD 外推至 0.6–2.3 Hz |
| III SPRINT 策略 | 本体观测 + 速度/角速度命令 | 残差 + 先验 → PD 目标，Isaac Gym PPO |

### 与相邻路线的对比（索引级）

| 维度 | SPRINT | AMP / SD-AMP | AI-CPG | Chasing Autonomy |
|------|--------|--------------|--------|------------------|
| 先验形态 | **可学习频谱生成器**（5 条参考） | 对抗判别器 + MoCap 分布 | 频域 CPG 模仿 | 动态重定向 + 控制引导 RL |
| 参考规模 | **5 clips** | 3–多 clip | 固定轨迹族 | 单演示扩展周期库 |
| 峰值速度（论文报告） | **6 m/s（G1 真机）** | SD-AMP ~3 m/s 快速模式 | < 4 m/s（仿真） | ~3.3 m/s（G1 户外） |
| 连续变速 | **0–6 m/s 单策略** | SD-AMP 速度条件 loco | 有限 | 目标条件 + 控制引导 |
| sim2real | **零样本（域随机 + AAC）** | SD-AMP 无额外微调叙述 | 主要仿真 | 户外验证 |

### 实验设置备忘

- **课程：** $v_x^{\mathrm{cmd}}\in[0,6]$ m/s，$f\in[0.6,2.3]$ Hz；$\omega_z^{\mathrm{cmd}}\in[-0.7,0.7]$ rad/s。
- **网络：** MLP 256 + LSTM 64；评价指标 $L_{\mathrm{rec}}, E_{\mathrm{BA}}, \mathrm{FID}, E_{\mathrm{qpos}}, E_{\mathrm{vel}}$。
- **跨平台：** 1.1 / 1.3 / 1.7 m 人形在 0.6–2.3 Hz 七档步态（慢走到冲刺）展示先验形态适应性（项目页视频）。

## 对 wiki 的映射

- 沉淀实体页：[SPRINT 人形竞技冲刺频谱先验（arXiv:2605.28549）](../../wiki/entities/paper-sprint-humanoid-athletic-sprints.md)
- 交叉补强：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[AMP & HumanX](../../wiki/methods/amp-reward.md)、[Motion Retargeting / GMR](../../wiki/methods/motion-retargeting-gmr.md)、[LAFAN1](../../wiki/entities/lafan1-dataset.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Chasing Autonomy Pipeline](../../wiki/methods/chasing-autonomy-pipeline.md)、[SD-AMP](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 当前提炼状态

- [x] 论文摘要与三阶段方法摘录
- [x] wiki 实体页映射确认
- [ ] 待公开非匿名项目页 / 代码仓库链接后补 `sources/sites/` 条目
