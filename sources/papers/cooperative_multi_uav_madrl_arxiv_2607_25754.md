# Cooperative Multi-UAV Navigation via Systematic MADRL（arXiv:2607.25754）

> 来源归档（ingest）

- **标题：** Cooperative Multi-UAV Navigation in Complex Environments via Systematic Multi-Agent Deep Reinforcement Learning
- **类型：** paper / multi-uav / cooperative-navigation / marl / madrl / airsim
- **arXiv：** <https://arxiv.org/abs/2607.25754>
- **PDF / HTML：** <https://arxiv.org/pdf/2607.25754> · <https://arxiv.org/html/2607.25754v1>
- **作者：** Yu Su、Nabil Aouf
- **机构：** City St George's University of London（School of Science and Technology）
- **提交：** 2026-07-28（13 pages, 7 figures）
- **项目页：** 无独立 `*.github.io` 或 lab 项目页（截至 2026-09-25）
- **代码：** arXiv 页与 HTML 全文 **未列出 GitHub / 数据链接** → 按 **确认未开源** 处理
- **入库日期：** 2026-09-25
- **一句话说明：** CTDE 式多智能体 SAC 框架：局部最优在线干预 + 分层协作示范缓冲与分级 BC + 成功率/碰撞率双条件课程 + LiDAR 结构域参数 \(\omega\) 驱动的 MoE  actor；AirSim/UE4 双机迷宫协作导航，maze_mix 零样本 \(\eta=0.75\)、含 1.5 m/s 横移动态障碍。

## 核心论文摘录（MVP）

### 1) 问题与总框架（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.25754>
- **核心贡献：** 复杂结构环境中多 UAV **协作到达各自目标** 面临：局部最优陷阱、稀疏协作奖励、智能体学习进度失衡、跨场景泛化不足。提出 **系统化 MADRL 框架**，四块机制：
  1. **局部最优感知与干预** — 访问记忆、方向新颖度、惩罚回传；执行层无额外可训练参数，触发时 override 动作；
  2. **分层协作示范缓冲 + 分级行为克隆** — 全队成功 vs 部分成功分池，场景/起终点配额；BC 直接监督 actor（非 GAIL/LfD 两阶段蒸馏）；
  3. **安全感知双条件课程** — 阶段推进需同时满足协作成功率 \(\eta\) 与碰撞率 \(\xi\) 阈值；回测已掌握场景 + replay 预填充防遗忘；
  4. **结构感知泛化** — 由 8 扇区 LiDAR 实时算 4 维域参数 \(\omega\)（死胡同/单墙/窄口/开阔），结构门控 **MoE** actor，减少对训练场景坐标的记忆。
- **对 wiki 的映射：**
  - [论文实体](../../wiki/entities/paper-sa-2607-25754-cooperative-multi-uav-navigation-madrl.md)
  - [强化学习](../../wiki/methods/reinforcement-learning.md)
  - [多旋翼仿真—规划—飞控栈总览](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

### 2) 观测、奖励与 CTDE 骨干（§II-B–D / Table I）

- **链接：** arXiv HTML Methodology
- **核心贡献：**
  - **仿真：** Microsoft **AirSim** + Unreal Engine 4；双 UAV；高度 1.8 m；动作 \(\mathbf{a}^i\in[-1,1]^3\) 速度指令；LiDAR 水平全向 12 m、8 扇区归一化距离 + 帧差。
  - **观测 33 维：** 目标方向/距离、速度、LiDAR、上一动作、访问记忆 4 维、\(\omega\) 4 维；历史长度 \(K\) 经 **LSTM** 编码。
  - **奖励：** 塑形距离（含目标遮挡时走廊约束 \(\kappa_t\) 防 reward vacuum）、访问/新颖度惩罚、分级到达奖励、静/动/机间碰撞项；**同步团队奖励** — 仅当所有智能体同时在目标邻域才给协作 bonus，并对到达时间差惩罚。
  - **优化：** **Multi-Agent SAC** + 集中式 critic；actor 为 **MoE**（共享 LSTM + \(\omega\) 门控 + \(N_e\) 专家头）。
- **对 wiki 的映射：**
  - [AirSim 实体](../../wiki/entities/airsim.md)
  - [quad-swarm-rl](../../wiki/entities/quad-swarm-rl.md) — 轻量 PyBullet MARL 对照

### 3) 基线对比 maze_05 / maze_mix（§IV-B / Table IV）

- **链接：** Results
- **核心贡献：** 各 1500 episode 训练；测试 **20 episode**。训练场景 **maze_05**；未见测试 **maze_mix**（含 **1.5 m/s** 横向往复 **动态障碍**）。
  - **完整框架：** maze_05 \(\eta=0.800\)、\(\xi=0.050\)；maze_mix 零样本 \(\eta=0.750\)、\(\xi=0.100\)、均终距 **3.260 m**；跨场景 \(\eta\) 落差 **0.050**。
  - **MAPPO：** maze_05 \(\eta=0.550\)、\(\xi=0.200\)；maze_mix \(\eta=0.300\)、\(\xi=0.350\)（泛化落差 **0.250**）；21 维观测（无 \(\omega\)）。
  - **Standard MASAC（去掉全部 proposed 机制）：** maze_05/ mix \(\eta\approx0.500/0.450\)，\(\xi\approx0.400\) — 高碰撞、低协作。
- **对 wiki 的映射：**
  - [CommNav](../../wiki/entities/paper-commnav.md) — 另一类多智能体导航（社会通信 vs 几何迷宫）
  - [FLAP](../../wiki/entities/paper-flap-fov-active-perception-3d-navigation.md) — 规划/优化路线 vs 本文 end-to-end RL

### 4) 消融与 MoE 泛化（§IV-C–D / Table V）

- **链接：** Ablation & Structure-aware Generalisation
- **核心贡献：**
  - **w/o intervention：** 前 ~800 ep \(\eta\approx0\)，\(\xi\) 长期 0.7–0.9，最终 \(\eta\approx0.40\)。
  - **w/o demo：** 长期稀疏成功，样本效率差；碰撞率反而偏低（保守探索）。
  - **w/o MoE：** maze_05 最终 \(\eta\approx0.70\)；**maze_mix 仅 \(\eta=0.250\)** — MoE+\(\omega\) 是跨场景关键。
  - **结论（§V）：** 当前 **双机**；仿真—真机鸿沟待填；未来扩展更大编队、真机验证、更广约束环境几何特征。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [EGO-Planner Swarm](../../wiki/entities/ego-planner-swarm.md) — 地图/ESDF 规划基线

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-sa-2607-25754-cooperative-multi-uav-navigation-madrl.md`](../../wiki/entities/paper-sa-2607-25754-cooperative-multi-uav-navigation-madrl.md)
- 互链：[AirSim](../../wiki/entities/airsim.md)、[multirotor 栈](../../wiki/overview/multirotor-simulation-planning-control-stack.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)、[quad-swarm-rl](../../wiki/entities/quad-swarm-rl.md)、[CommNav](../../wiki/entities/paper-commnav.md)、[FLAP](../../wiki/entities/paper-flap-fov-active-perception-3d-navigation.md)

## BibTeX

```bibtex
@article{su2026cooperative,
  title   = {Cooperative Multi-{UAV} Navigation in Complex Environments via Systematic Multi-Agent Deep Reinforcement Learning},
  author  = {Su, Yu and Aouf, Nabil},
  journal = {arXiv preprint arXiv:2607.25754},
  year    = {2026}
}
```
