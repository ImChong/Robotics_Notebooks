# MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots（arXiv:2605.13058）

> 来源归档（ingest）

- **标题：** MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots
- **类型：** paper / wheeled-legged quadruped / blind locomotion / multi-skill RL / constrained RL / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2605.13058>
- **PDF：** <https://arxiv.org/pdf/2605.13058>
- **项目页：** <https://hyzenthlayer.github.io/mujica/>
- **会议：** IEEE ICRA 2026
- **作者：** Yuqi Li, Peng Zhai*, Yueqi Zhang, Xiaoyi Wei, Quancheng Qian, Zhengxu He, Qianxiang Yu, Lihua Zhang*（复旦大学智能机器人与先进制造学院等）
- **硬件：** Unitree **Go2-W** 轮足四足；机载 **NVIDIA Jetson Orin NX**；仿真 **Isaac Lab**（4096 并行环境，RTX 4090）
- **入库日期：** 2026-06-19
- **一句话说明：** 仅用本体感知的 **统一轮足策略** 联合学习全向移动、高台攀爬与摔倒恢复三类高难度技能，并以 **速度相关 DC 电机硬约束（P3O）** 与 **高层技能选择器** 实现安全 sim2real 与自主模态切换；真机验证 **1 m 室内高台** 与连续多技能任务。

## 摘要级要点

- **问题：** 轮足机器人需同时协调 **轮式驱动** 与 **足式控制**；纯本体感知下多技能盲控往往技能单一或需人工切换；多数 RL 方法对 **DC 电机速度–扭矩包络** 建模过简，限制极限机动与安全部署。
- **方法 MUJICA（Multi-skill Unified Joint Integration of Control Architecture）：**
  - **C-POMDP + 非对称 Actor–Critic**：Actor 仅本体；Critic 用特权观测；约束优化采用 **P3O**（reward critic + constraint critic）。
  - **状态估计器（GRU，H=6）**：联合预测基座线速度、轮–地距离、机身分段碰撞概率与隐变量（对比学习 SwAV，受 HIM 启发）。
  - **多任务统一策略**：技能指示变量 $\zeta_t$ 解耦观测；三技能——(i) 全向速度跟踪、(ii) 高台攀爬（仅线速度）、(iii) 摔倒恢复（重力/站立姿态奖励）。
  - **DC-motor 硬约束**：按 Unitree 电机手册建模 **低速恒扭矩、高速线性降额**；小腿关节还受 **位置相关余弦扭矩上限**。
  - **两阶段训练：** S1 固定 $\zeta$ 训练低层技能（30k iter）；S2 冻结低层，训练 **仅本体的高层技能选择器**（10k iter），统一速度跟踪奖励。
  - **课程学习：** $33\times 20$ 网格多任务地形（楼梯、坡、离散障碍、粗糙地、坑洞等），难度随成功/失败自适应。
- **仿真对比：** 相对 DreamWaQ+P3O、Vanilla PPO 及三项估计消融，MUJICA 在 5 类代表任务 × 10 难度上成功率全面领先；无 DC 约束时大腿扭矩违规 **>90%**，有约束后 **<3.5%**。
- **连续任务（TABLE III，难度 5）：** 楼梯恢复 → 楼梯攀爬 → 高台攀爬链式任务：无 indicator（共享奖励）崩溃；无 indicator（分奖励）38% 高台成功率；**技能选择器 91%**。
- **真机：** 零样本 sim2real；30° 楼梯摔倒恢复、20 cm 不规则楼梯、**80 cm 户外 / 100 cm 室内高台**；随机倒地后自动恢复→全向移动→60 cm 高台的无干预连续演示。

## 核心摘录（面向 wiki 编译）

### 与相关路线的关系

| 维度 | MUJICA | MoE / 蒸馏多技能 | DreamWaQ + P3O | 轮足盲走 MOE（Zhang 2024） |
|------|--------|------------------|----------------|---------------------------|
| 技能整合 | **单策略 + $\zeta$ 指示 + 高层选择器** | 多专家加权或蒸馏 | 单任务估计 + 约束 | 混合专家门控 |
| 感知 | **纯本体（盲）** | 多为本体 | 本体 + 隐式地形 | 本体 |
| 安全 | **DC 电机硬约束（P3O）** | 各异 | P3O 约束 | 未强调电机包络 |
| 平台 | **Go2-W 轮足** | 多四足 | 四足 | 轮足四足 |
| 难点技能 | **高台攀爬 + 摔倒恢复 + 全向** | 步态切换为主 | 平地/楼梯类 | 盲走 |

### 训练与部署参数（摘录）

| 项目 | 设置 |
|------|------|
| 控制频率 | 50 Hz（低层 + 选择器） |
| 仿真步长 | 200 Hz 物理 |
| 动作空间 | 腿关节：相对默认姿态的角度偏移 + PD；轮关节：期望角速度 + 阻尼 |
| 域随机化 | 摩擦、质量偏置、外力、推搡、电机增益等（TABLE II） |
| 基线 | DreamWaQ+P3O、Vanilla PPO；消融：去速度/轮高/碰撞估计 |

### 对 wiki 的映射

- 沉淀实体页：[MUJICA（arXiv:2605.13058）](../../wiki/entities/paper-mujica-wheel-legged-multi-skill.md)
- 交叉补强：[轮足四足机器人](../../wiki/concepts/wheel-legged-quadruped.md)、[Hybrid Locomotion](../../wiki/tasks/hybrid-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
