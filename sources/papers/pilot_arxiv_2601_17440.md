# PILOT: A Perceptive Integrated Low-level Controller for Loco-manipulation over Unstructured Scenes（arXiv:2601.17440）

> 来源归档（ingest）

- **标题：** PILOT: A Perceptive Integrated Low-level Controller for Loco-manipulation over Unstructured Scenes
- **类型：** paper / humanoid / loco-manipulation / perceptive locomotion / whole-body RL / elevation map
- **arXiv abs：** <https://arxiv.org/abs/2601.17440>
- **arXiv HTML：** <https://arxiv.org/html/2601.17440>
- **PDF：** <https://arxiv.org/pdf/2601.17440>
- **机构：** 上海交通大学自动化与智能感知学院、上海交通大学全球学院、上海交通大学计算机学院；通讯作者 Hesheng Wang（wanghesheng@sjtu.edu.cn）
- **硬件：** Unitree G1（29 DoF）；实机 **LiDAR 机器人系高程图**
- **仿真：** NVIDIA Isaac Lab；单卡 NVIDIA RTX 4090
- **入库日期：** 2026-06-05
- **一句话说明：** **单阶段 PPO** 统一 **感知行走 + 大工作空间全身操作**：跨模态编码器融合 **预测式本体历史** 与 **注意力多尺度 11×11 高程图**；**4 专家 MoE** 协调运动/操作模式；**渐进命令课程**（无 MoCap 偏置）；G1 真机楼梯/高台等 **非结构化 loco-manipulation** 遥操作与分层 RL 自主任务。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 相邻 loco-manipulation | [WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA) README | OpenDriveLab 将 PILOT 列为相关 arXiv:2601.17440 工作（统一全身 loco-manipulation 另一路线） |
| 感知行走先验 | PIM（论文 [10]） | 本体历史预测 + 混合内部模型 + LiDAR 高程图；PILOT  proprio 编码器 **沿用该框架** |
| 注意力高程图 | He et al. [9] | CNN + 注意力地图编码；PILOT 外感知 **本体引导 cross-attention** 受其启发 |
| 解耦基线 | HOMIE [14]、FALCON [21]、AMO [12] | 简单地形对比；**无感知** 故未参与全地形评测 |
| 统一无感知 | ULC [13]、MoCap 模仿 [11,15] | 论文对比的 **盲走/分布偏置** 路线 |

> 截至入库时，**未见论文作者公开的项目页或代码仓库**；后续若发布可补 `sources/sites/` 或 `sources/repos/`。

## 摘要级要点

- **问题 1：** 多数全身低层控制器 **无外感知**，在非平面场景（楼梯、崎岖地面）上边移动边操作时易绊脚、上肢不稳。
- **问题 2：** **统一全身策略** 维数高、目标冲突（平衡 vs 末端精度）；MoCap 引导易 **运动学偏置**；再加感知会加剧梯度干扰。
- **PILOT 回答：** 单策略端到端 **29 维动作 → PD**；**跨模态上下文编码**（本体预测 $\hat v_t,\hat z_{t+1}$ + 多尺度注意力高程图 $z_t^p$）；**MoE（N=4）** + 二元模式 $I_t$（自然行走 vs 上肢跟踪）；上肢 **残差** 叠加 $q_t^{\mathrm{upper}^*}$。
- **感知：** $o_t^p\in\mathbb{R}^{121}$，**11×11、0.1 m** 栅格，每格为地形相对 base 的竖直距离；实机 **LiDAR 高程图**。
- **命令：** $c_t=[v^x,v^y,\omega^{\mathrm{yaw}},h^{\mathrm{base}},\mathbf{rpy},q^{\mathrm{upper}^*}]$；地形含上下楼梯、坡、高台、随机崎岖。
- **训练课程：** 先纯 locomotion → 基座高度 → 躯干姿态 → 上肢指数采样命令（借鉴 HOMIE）；**不用 MoCap**。
- **部署：** **VR 头显 + 手柄** 长程遥操作；另 **分层 RL** 自主（导航 + 弯腰取箱等）。
- **仿真（简单地形 vs 基线）：** 跟踪误差普遍低于 HOMIE / FALCON / AMO / PILOT w/o vision（Table IIa）。
- **消融（全地形）：** 完整 PILOT  stumble $E_{\mathrm{stumble}}=0.006$；去视觉 0.087、去注意力 0.066、去 MoE 0.017。

## 核心摘录（面向 wiki 编译）

### 1) 跨模态编码器

| 分支 | 输入 | 输出 / 监督 |
|------|------|-------------|
| 预测式本体 | $o_{t-H:t}^n$（$H=5$）+ 当前 $o_t^p$ | $\hat v_t$（MSE）、$\hat z_{t+1}$（对比学习，对齐 future encoder） |
| 注意力外感知 | 11×11 高程图 | 全局 MLP $\phi(o^p)$ + PointNet 式局部特征；**$o^n$ 为 Query** 的 MHA → $z_t^p$ |

### 2) MoE 全身策略

- 输入：$\{z_t^p, z_t^o, I_t\}$，$z_t^o$ 聚合 $z_t^H, v_t, o_t^n$。
- $I_t=0$：臂参考固定名义姿态，优先协调 locomotion；$I_t=1$：跟踪采样 $q^{\mathrm{upper}^*}$。
- $a_t=\sum_{i=1}^4 p_i^t a_i^t$；$a^{\mathrm{upper}}$ 为对 $q^{\mathrm{upper}^*}$ 的 **残差修正**。

### 3) 奖励（Table I 摘要）

- **任务：** 线/角速度、基座高度、躯干姿态、上肢位置跟踪；终止 $-200$。
- **正则：** action rate、能耗、关节偏离、**feet stumble**、air time、滑移、大足力、no fly。

### 4) 与仓库内路线的关系

| 维度 | PILOT | Explicit Stair Geometry | FastStair | VIRAL / DoorMan |
|------|-------|-------------------------|-----------|-----------------|
| 任务 | **loco-manipulation LLC** | 楼梯 locomotion | 楼梯高速上楼 | 高层 + 预训练 WBC |
| 感知 | 11×11 高程 + cross-attn | 4D 几何 token | 机载高程 + DCM 落点 | RGB / 视觉 Sim2Real |
| 全身 | **单阶段统一 29DoF** | 下身为主 | 下身 + 规划 | delta 至 HOMIE 类底层 |
| 平台 | **G1** | G1 | LimX Oli | G1 |

## 对 wiki 的映射

- 沉淀实体页：[PILOT 感知统一 loco-manipulation 低层控制器（arXiv:2601.17440）](../../wiki/entities/paper-pilot-perceptive-loco-manipulation.md)
- 交叉补强：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[楼梯与障碍中心节点](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)

## 当前提炼状态

- [x] 摘要、方法与实验要点摘录
- [x] wiki 实体页与任务页交叉链接规划
- [ ] 待作者公开项目页 / 代码后补 `sources/sites/` 或 `sources/repos/`
