# LadderMan: Learning Humanoid Perceptive Ladder Climbing（arXiv:2606.05873）

> 来源归档（ingest）

- **标题：** LadderMan: Learning Humanoid Perceptive Ladder Climbing
- **类型：** paper / humanoid / perceptive locomotion / ladder climbing / loco-manipulation / depth / sim2real / teacher-student
- **arXiv abs：** <https://arxiv.org/abs/2606.05873>
- **arXiv HTML：** <https://arxiv.org/html/2606.05873>
- **PDF：** <https://arxiv.org/pdf/2606.05873>
- **项目页：** <https://ladderman-robot.github.io/>（归档见 [`sources/sites/ladderman-robot-github-io.md`](../sites/ladderman-robot-github-io.md)）
- **机构：** Amazon FAR、USC、UC Berkeley、Stanford University、CMU；† Amazon FAR team co-lead
- **作者：** Siheng Zhao, Yuanhang Zhang, Ziqi Lu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Yue Wang, C. Karen Liu, Guanya Shi
- **硬件：** Unitree G1（1.3 m、29-DoF）；机载 **Intel RealSense D435i**；机载 **NVIDIA Jetson Orin**
- **仿真：** NVIDIA **IsaacSim**；深度渲染 **NVIDIA Warp**；专家策略 **PPO**；统一策略 **DAgger + PPO + KL 到专家**
- **参考动作：** 单条 OptiTrack 动捕参考（梯子 A：倾角 $\phi=65.5°$，踏棍间距 $z=24.8$ cm）
- **入库日期：** 2026-06-09
- **一句话说明：** **两阶段** 人形 **梯子攀爬 + 梯上操作** 系统：Stage 1 用 **hybrid motion tracking**（上身松弛跟踪 + 梯子中心接触/攀爬奖励）从 **单条参考动作** 学多几何 **专家**；Stage 2 以 **hybrid DAgger + RL** 蒸馏为 **深度 visuomotor 统一攀爬策略**；真机深度经 **VFM（Fast-FoundationStereo）** 与 **rung-focused masking（RFM）** 桥接 sim-to-real；梯上 manipulation 用 **双智能体**（下身稳定 + 上身遥操作）解耦，相对 TWIST2 更稳。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://ladderman-robot.github.io/> | 双向攀爬、梯上操作、VFM/RFM 消融、仿真成功率热力图 |
| 姊妹多向深度行走 | [RPL（arXiv:2602.03002）](https://arxiv.org/abs/2602.03002) | 同 Amazon FAR 系；RPL 侧重 **双向楼梯/坡/垫脚石 + 载荷**，LadderMan 侧重 **梯子稀疏踏棍 + 梯上操作** |
| 姊妹感知跑酷 | [PHP（arXiv:2602.15827）](https://arxiv.org/abs/2602.15827) | 同系深度感知人形；PHP 强调 **动态跑酷技能链**，LadderMan 强调 **细粒度多接触攀爬** |
| 感知 loco-manipulation LLC | [PILOT（arXiv:2601.17440）](https://arxiv.org/abs/2601.17440) | 单阶段 LiDAR 高程 **全身 LLC**；LadderMan 攀爬与操作 **分策略 + 双智能体** |
| 全身解耦基线 | FALCON [36]、TWIST2 [34] | LadderMan 梯上操作相对 **现成全身遥操作** 更稳 |
| 运动跟踪基线 | BeyondMimic [9]、DeepMimic [17] | 盲 motion tracking 在 OOD 梯子几何上近零成功率 |
| VFM 深度 | Fast-FoundationStereo [27] | 真机立体深度估计，替代重度 depth randomization |

> 截至入库时，论文承诺 **训练/推理代码与可部署模型将开源**；尚未见作者公开仓库链接。

## 摘要级要点

- **问题 1：** 梯子攀爬需 **稀疏窄踏棍/扶手上的全身手–足多接触协调**；感知与控制误差易灾难性失稳。
- **问题 2：** 既往梯子工作多 **模型驱动、无感知闭环、任务特调硬件**；难泛化到多样倾角与踏棍间距。
- **问题 3：** 感知人形行走多 **下身主导**；梯子与梯上操作需 **全身协调** 与 **稳定平衡下的上肢交互**。
- **LadderMan 回答：**
  - **Stage 1 专家：** **Hybrid motion tracking** — 上身关键点半权重跟踪、下身全权重；加 **梯子中心接触奖励** $r^{\text{contact}}$ 与 **攀爬任务奖励** $r^{\text{task}}$；相对 DeepMimic 可从 **单参考** 学到 $(55°,20\text{cm})$–$(70°,30\text{cm})$ 多几何专家。
  - **Stage 2 统一策略：** **DAgger + PPO**，损失 $L = L_{\text{RL}} + \lambda D_{\text{KL}}(\pi^{\text{visual}} \| \pi^{\text{expert}})$，$\lambda$ 退火；输入 **本体 + 深度 + 二值攀爬方向**；Warp 高吞吐深度渲染。
  - **Sim2Real 深度：** 真机用 **VFM** 得几何一致深度；训练侧 **轻量噪声**（$\mathcal{N}(0,0.005)$ + 5% dropout）+ **RFM**（$p=0.1$ 遮非踏棍区）聚焦任务相关结构。
  - **梯上操作：** **双智能体** $\pi^{\text{manip}}=[\pi^u;\pi^l]$ — 下身维持接触与骨盆姿态，上身跟踪 VR（PICO 4 Ultra）遥操作目标；相对 TWIST2 切换后不易失稳。
- **真机：** 梯子 A/B/C 零样本攀爬；双向约 **20 s** 完成；攀爬速度 **~3.4 s/踏棍**（人类 ~3.2 s）；真机消融：w/o RL **2/10**，w/o VFM **3/10**，w/o RFM **0/10**（梯子 A，10 次随机初姿）。

## 核心摘录（面向 wiki 编译）

### 1) Hybrid motion tracking 奖励（相对标准 tracking）

- 上身 $K_u$、下身 $K_l$ **非对称指数跟踪**（式 1）：上身半权重、下身全权重，允许上身按梯子几何自适应接触。
- **接触奖励**（式 2）：参考接触指示 $\hat{c}$ + 末端位置 $P$ 对梯子几何自动生成的目标 $\hat{P}$。
- **任务奖励**：根位姿 $R$ 对目标 $\mathcal{G}$ 的 $L_2$ 距离。
- **终止**：标准 tracking 误差阈值 + **梯子接触丢失 >30 帧**。
- **域随机化**：梯子 $(\phi,z)$ 局部扰动、踏棍几何/摩擦随机化。

### 2) 统一 visuomotor 策略

- 观测：本体、深度、**向上/向下** 二值指令。
- 训练深度：**NVIDIA Warp** 渲染；clip $[0.1, 2]$ m。
- 纯 DAgger 不足：专家未覆盖全配置空间 + 模仿误差在动态接触任务中累积 → **必须加 RL**。

### 3) Sim2Real 感知组件

| 组件 | 作用 |
|------|------|
| **VFM** | Fast-FoundationStereo 真机深度；减轻传感器缺失像素/噪声 |
| **RFM** | 训练时 10% 概率遮非踏棍区，减少对扶手/背景的过拟合 |
| **轻量随机化** | 高斯噪声 + dropout；避免重度 hand-tuned depth DR |

### 4) 双智能体梯上操作

| 智能体 | 输入 | 奖励 |
|--------|------|------|
| $\pi^l$ | 全身本体 $s_t$ | 接触 $r^{\text{contact}}$ + 稳定 $r^{\text{task}}$ |
| $\pi^u$ | $s_t$ + 遥操作上肢目标 $g_t$ | 上身 tracking（式 1 上身项） |
| 训练上肢目标 | AMASS 随机采样 | 部署：IK 自 PICO 4 Ultra |

### 5) 实验摘要

| 设置 | 要点 |
|------|------|
| 仿真网格 | $\phi \in \{55°,60°,65°,70°\}$；$z \in \{20,22,24,26,28,30\}$ cm；每格 100 episode |
| 仿真成功率 | 踏棍 24–28 cm、倾角 55–65° 区间 **>95%**；边界配置平均 **~57%** |
| 盲基线 | BeyondMimic 式无感知 tracking：峰值 **49%**（近参考几何），OOD 近零 |
| 真机梯子 | A $(65.5°,24.8\text{cm})$、B $(69.2°,23.1\text{cm})$、C $(64.8°,25.5\text{cm})$ |
| 真机成功率（全文） | Ours：A **9/10**，B **6/10**，C **7/10** |

### 6) 与仓库内路线的关系

| 维度 | LadderMan | RPL | PHP | FastStair | PILOT |
|------|-----------|-----|-----|-----------|-------|
| 任务 | **梯子攀爬 + 梯上操作** | 双向多地形行走 + 载荷 | 跑酷技能链 | 高速上楼梯 | loco-manipulation LLC |
| 结构 | **稀疏踏棍/扶手** | 楼梯/坡/垫脚石 | 动态障碍 | 重复踢面踏面 | 非结构化地形 |
| 感知 | **单深度 + VFM** | 多视角深度 | 单深度 | 机载高程 | LiDAR 高程 |
| 训练 | **单参考 hybrid tracking 专家 + DAgger+RL** | 分地形高程专家 + DAgger | MM 参考 + DAgger+PPO | DCM + 分速专家 | 单阶段 MoE PPO |
| 操作 | **梯上双智能体遥操作** | 2 kg 载荷行走 | — | — | 全身 LLC |

## 对 wiki 的映射

- 沉淀实体页：[LadderMan 人形感知梯子攀爬（arXiv:2606.05873）](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)

## 当前提炼状态

- [x] 摘要、两阶段管线、VFM/RFM、双智能体操作与实验要点摘录
- [x] wiki 实体页与任务页交叉链接规划
- [ ] 待作者公开代码后补 `sources/repos/`
