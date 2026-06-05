# Learning Humanoid Standing-up Control across Diverse Postures（arXiv:2502.08378）

> 来源归档（ingest）

- **标题：** Learning Humanoid Standing-up Control across Diverse Postures
- **简称：** HoST（Humanoid Standing-up Control）
- **类型：** paper / humanoid / whole-body RL / sim2real / fall recovery
- **arXiv abs：** <https://arxiv.org/abs/2502.08378>
- **arXiv HTML：** <https://arxiv.org/html/2502.08378v1>
- **PDF：** <https://arxiv.org/pdf/2502.08378>
- **项目页：** <https://taohuang13.github.io/humanoid-standingup.github.io/>
- **代码：** <https://github.com/InternRobotics/HoST>（README 亦引用 <https://github.com/OpenRobotLab/HoST>，同一实现）
- **视频：** [YouTube](https://www.youtube.com/watch?v=Yruh-3CFwE4) · [Bilibili](https://www.bilibili.com/video/BV1o2KPeUEob/)
- **机构：** Shanghai AI Laboratory；SJTU；HKU；ZJU；CUHK
- **会议：** RSS 2025（**Best Systems Paper Finalist**）
- **硬件：** Unitree G1（23-DoF，真机直接部署）；扩展支持 Unitree H1、High Torque Mini Pi、DroidUp（2025-05/06 新闻）
- **入库日期：** 2026-06-05
- **一句话说明：** 无预定义轨迹、从零 PPO 学习人形**多姿态起身**；**多 critic + 四地形课程 + 竖直拉力探索 + L2C2 平滑与动作 rescaler 约束**，Isaac Gym 训练后 **G1 真机零额外 sim2real 微调** 部署。

## 摘要级要点

- **问题：** 起身控制是人形跌倒恢复与「坐/躺→站立」场景的基础能力；既有方法多限于仿真、依赖地面特定参考轨迹，或无法在沙发/斜坡/户外等多姿态真机泛化。
- **方法（HoST）：** 将起身建模为 MDP，**本体感知状态**（IMU 角速度/roll/pitch、关节位速、上步动作、动作缩放系数 $\beta$）+ **5 帧历史**；动作为关节位置增量经 PD 力矩执行；**四组奖励**（task / style / regu / post）按三阶段（扶正→跪姿→站起）激活；**多 critic PPO** 对各奖励组独立估计 return 再加权 advantage。
- **探索：** 训练早期对基座施加**竖直拉力课程** $\mathcal{F}$（躯干近竖直后生效，随 episode 末高度目标递减），类比婴儿扶起辅助。
- **真机约束：** **动作 rescaler** $\beta$ 课程收紧输出界以抑制暴力冲撞；**L2C2** 平滑正则抑制振荡。
- **Sim2Real：** **四仿真地形**（ground / platform / wall / slope）覆盖多样初始姿态 + **域随机化**（质量、CoM、摩擦、PD 增益、力矩 RFI、控制延迟等）；**4096** 并行环境，Isaac Gym + rsl_rl PPO。
- **评测指标（仿真协议）：** $E_{\mathrm{succ}}$、$E_{\mathrm{feet}}$（脚移动距离）、$E_{\mathrm{smth}}$、$E_{\mathrm{engy}}$；真机展示抗外力、绊脚物、**12 kg** 载荷、极端初始姿态等。

## 与相关路线的对比（论文 Table I 归纳）

| 能力维度 | 典型轨迹跟踪 RL | 地面参考 + RL | 从零 RL（仿真） | 真机 + 无参考 | 超地面姿态 | 高 DoF | 单阶段训练 |
|----------|----------------|---------------|-----------------|---------------|------------|--------|------------|
| HoST | — | — | — | ✓ | ✓ | ✓ | ✓ |

与 [SD-AMP](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md) 的差异：**SD-AMP** 用 **AMP 运动先验 + 重力门控双判别器** 在**行走/跑/跌倒起身**间统一；**HoST** 专注**纯起身技能**、**无 MoCap 参考**、强调**非地面初始姿态**（墙/台/坡）与**系统级真机约束**（平滑/速度界），RSS 2025 系统论文 finalist。

## 核心摘录（面向 wiki 编译）

### 三阶段与奖励分组

1. **扶正（righting）** → 2. **跪姿（kneeling，$h_{\mathrm{base}}$ 阈值）** → 3. **站起（rising）**；各阶段激活不同 task/style/regu/post 奖励子集。

### 观测与动作接口

- 状态 $s_t=[\omega_t, r_t, q_t, p_t, \dot{p}_t, a_{t-1}, \beta]$；策略 50 Hz，仿真 PD 200 Hz、真机 PD 500 Hz。
- PD 目标 $p_t^d = p_t + \beta a_t$，$a_t\in[-1,1]^{|A|}$。

### 消融要点（仿真，节选）

- **无多 critic（w/o MuC）：** 四地形成功率均 **0%**。
- **无竖直拉力（w/o Force）：** ground/wall/slope 多为 0%，platform 略可学。
- **无动作界（w/o Bound）：** 成功率仍高但 $E_{\mathrm{feet}}$、$E_{\mathrm{smth}}$、$E_{\mathrm{engy}}$ 显著恶化（暴力动作）。
- **History-5** 为默认；过短历史损害 wall/slope 等难地形。

## 对 wiki 的映射

- 沉淀实体页：[HoST 人形多姿态起身（arXiv:2502.08378）](../../wiki/entities/paper-host-humanoid-standingup.md)
- 交叉补强：[Balance Recovery](../../wiki/tasks/balance-recovery.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[SD-AMP](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)
- 原始配套：[项目页](../sites/host-humanoid-standingup-project.md)、[代码仓库](../repos/host_internrobotics.md)

## 参考来源（原始）

- Huang et al., arXiv:2502.08378, 2025
- 项目页与演示视频（见上链接）
