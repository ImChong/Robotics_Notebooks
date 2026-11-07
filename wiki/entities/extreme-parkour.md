---

type: entity
tags:
  - quadruped
  - reinforcement-learning
  - locomotion
  - perception
  - sim2real
  - privileged-training
  - parkour
status: complete
updated: 2026-11-07
arxiv: "2309.14341"
venue: "ICRA 2024"
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../concepts/privileged-training.md
  - ../concepts/sim2real.md
  - ../concepts/terrain-adaptation.md
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ./legged-gym.md
  - ./unitree.md
  - ./dreamwaq-plus.md
  - ./paper-swap-parkour.md
  - ./paper-vision-locomotion-egocentric.md
  - ../methods/dagger.md
original:
  - ../../sources/papers/extreme_parkour_arxiv_2309_14341.md
sources:
  - ../../sources/papers/extreme_parkour_arxiv_2309_14341.md
  - ../../sources/repos/extreme-parkour.md
  - ../../sources/sites/extreme-parkour-github-io.md
summary: "Extreme Parkour（ICRA 2024）：Unitree A1 低成本四足在单目深度下端到端跑酷；两阶段 RL（特权 scandots + oracle 航向 → 深度 + 自预测 yaw 双重 DAgger 蒸馏），约 20 小时仿真训练。"
tags: [quadruped, reinforcement-learning, locomotion, perception, sim2real, privileged-training, parkour, unitree]

---

# Extreme Parkour（端到端四足感知跑酷）

**Extreme Parkour**（Cheng et al., [arXiv:2309.14341](https://arxiv.org/abs/2309.14341)，**ICRA 2024**）在 **Unitree A1 低成本四足**（论文 §4.1；并非 Go1） 上，用 **单目前向深度相机 + 大规模仿真 RL**，训练 **单神经网络** 直接输出关节控制，实现高跳、远跳、手倒立与斜 ramp 自主变向等 **极限跑酷** 技能。官方材料：[项目页](https://extreme-parkour.github.io/)、[GitHub](https://github.com/chengxuxin/extreme-parkour)。

## 一句话定义

**Phase 1 用特权 scandots 与 terrain waypoint 航向训 PPO Teacher，Phase 2 用 DAgger 把外感知与 yaw 决策蒸馏进 ConvNet–GRU 深度 Student——部署时仅深度图即可输出精确跑酷控制。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| DAgger | Dataset Aggregation | 迭代收集策略诱导状态下的专家标注以纠偏的模仿学习方法 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| GRU | Gated Recurrent Unit | 门控循环单元，处理时序观测 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| RMA | Rapid Motor Adaptation | 从历史轨迹隐式估计环境参数的快速运动自适应 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

- **端到端视觉 loco 的早期标杆：** 在 **执行器不精确、深度低频抖动** 的真实约束下，仍实现 **高跳 0.5 m（约 2× 大腿关节高度 26 cm）、远跳 0.8 m（约 2× 前后足间距）** 级动态动作（论文 §4.2；Table 1 按「相对机身高度 / 体长 2×」比较，同期工作为 1.5×），影响后续 Robot Parkour Learning、DreamWaQ++ 等感知 locomotion 叙事。
- **双重蒸馏范式清晰：** 不仅蒸馏 **motor commands**，还蒸馏 **rapidly fluctuating heading**（ramp 跳上角与即时变向人类无法遥操作给 waypoint）——比「只换 scandots→深度」更完整。
- **训练成本可接受：** 论文 §4.1 与 README 均称单张 **3090 上 < 20 h** 可训出可部署策略，降低复现门槛；开源栈基于 **legged_gym + Isaac Gym**。
- **奖励设计可迁移：** **Inner-product 航向跟踪 + clearance 边缘惩罚** 解决跑酷中「绕障 exploit」与「贴边省能」两类典型失败（项目页 ablation）。

## 核心结构

| 模块 | 作用 |
|------|------|
| **Phase 1 Teacher（PPO）** | 输入本体 $x$、特权 **scandots** $m$、oracle 航向 $\hat{d}$（waypoint 计算）、行走标志与速度指令；**ROA** 从历史估计环境参数。 |
| **统一奖励** | $\hat{d}_w=(p-x)/\|p-x\|$ 与速度跟踪对齐；**$r_{\mathrm{clearance}}$** 惩罚足端距地形边缘 < 5 cm 的接触。 |
| **Phase 2 Student（DAgger）** | (1) **ConvNet–GRU** 深度管线替代 scandots（RMA 式）；(2) **从深度自预测 yaw**，替代人类 waypoint。 |
| **部署** | `save_jit.py` 导出 traced 模型；实机 **零微调** 叙事（论文 / 项目页设定）。 |

### 流程总览

```mermaid
flowchart TB
  subgraph p1["Phase 1 — 特权 RL（PPO）"]
    SC["Scandots m + 本体 x"]
    WP["Terrain waypoint → 航向 d̂"]
    ROA["ROA 环境估计"]
    TEACHER["Teacher π_base"]
    SC --> TEACHER
    WP --> TEACHER
    ROA --> TEACHER
  end
  subgraph p2["Phase 2 — 双重 DAgger 蒸馏"]
    DEP["前向深度图"]
    CNN["ConvNet–GRU"]
    YAW["航向预测头"]
    STUDENT["Student π_deploy"]
    DEP --> CNN
    DEP --> YAW
    CNN --> STUDENT
    YAW --> STUDENT
  end
  TEACHER -.->|DAgger 标签| STUDENT
  STUDENT --> ACT["关节控制 → 实机 A1"]
```

## 方法栈（提炼）

- **与分模块 classical stack 对比：** 传统跑酷分别精密标定感知 / 执行 / 控制；本文 **单 NN + 仿真 RL** 在 **不改造硬件** 前提下习得多样技能。
- **Teacher–Student 谱系：** 与 [RMA](../concepts/privileged-training.md)、Robot Parkour Learning（ZiwenZhuang）同族；Extreme Parkour 特色是 **航向蒸馏** 与 **clearance 奖励** 针对跑酷几何。
- **与 DreamWaQ++ 差异：** DreamWaQ++ 强调 **点云多模态 + 非对称 AC 单阶段** 与传感器失效回退；Extreme Parkour 强调 **极限动态技能 + 深度端到端 + 两阶段蒸馏**（见 [DreamWaQ++](./dreamwaq-plus.md) 对比表）。

## 实验与 ablation（论文 §4 + 项目页）

- **技能：** High jump（0.5 m 箱）、Long jump（0.8 m gap）、Handstand（无外感知，仅本体即可在草地与下楼梯行走）、37° 倾斜 ramp、Step / Gap / Hurdle 组合；对新障碍组合与不同物理属性泛化。
- **仿真奖励消融（论文表 2，256 机器人 / 30 s，总计 MXD↑ / MEV↓）：** Ours 0.98 / 0.03；NoInner 0.75 / 0.04（Step 地形 MXD 仅 0.14，只会撞台阶重试，Hurdle 上学会绕行故 MEV 低）；NoClear 0.99 / 0.08（MXD 略高但贴边落脚，实机不稳）；Noisy（模拟融合高程图噪声）0.82 / 0.20，方差大。
- **蒸馏消融（表 3）：** Ours 0.92 / 0.09，几乎等于 Oracle 的 0.94 / 0.10；Both（始终观测预测 yaw）0.12 / 0.26、Mask（yaw 置零）0.05 / 0.00，均因噪声 yaw 导致模仿数据分布漂移而不收敛。
- **实机（图 7）：** 每地形每难度 5 次试验，最难一档成功率比 NoClear / NoDir 高 20–80%；NoDir（人用摇杆给方向）在跳跃与斜坡处需要突然变向，超出训练分布而失败。
- **Vision Locomotion 对照：** 同团队 [CoRL 2022 ego-depth loco](./paper-vision-locomotion-egocentric.md) 在项目页的楼梯场景展示跌落（项目页材料，非论文表格）。

## 工程实践

| 项目 | 论文 / README 给出的设定 |
|------|--------------------------|
| 机器人 | Unitree A1，12 关节；站立大腿关节高 26 cm，体长 40 cm |
| 感知硬件 | 头部 Intel RealSense D435，深度 10 ± 2 Hz；裁左侧坏像素并降采样至 58 × 87 |
| 机载计算 | Jetson NX；深度骨干 10 Hz、基策略 50 Hz，UDP 通信 |
| 延迟对齐 | 深度延迟强制补足到 0.08 s、本体延迟固定 0.016 s，用固定延迟抑制抖动（论文 §4.1；README 训练命令里 `--delay` 在 8k 迭代后启用延迟） |
| 训练预算 | 单张 3090 < 20 h；README：Base 10–15k iter、Distillation 5–10k iter |
| 课程 | 走过超过一半地形长度升级，走不到 $v_{cmd}T$ 一半降级 |
| Phase 2 | DAgger，学生动作步进环境；actor 由 Phase 1 拷贝初始化；MTS：预测与 oracle 的 yaw 差小于 0.6 才用预测 yaw，否则用 oracle |
| 部署 | `save_jit.py` 导出 traced 模型到 `logs/parkour_new/<exptid>/traced/`（README） |

调参提示：先看 MEV（贴边次数）而非只看位移；若策略在 Hurdle 上绕行，优先检查跟踪奖励是否误用机体系速度；若 Phase 2 不收敛，先检查是否直接使用了预测 yaw（应走 MTS）。

## 常见误区或局限

- **误区：** 把 Extreme Parkour 等同于「又一个 legged_gym 走路」——其核心是 **parkour 地形 + 统一奖励 + 双重蒸馏**，普通 velocity tracking 奖励不能直接迁移。
- **误区：** 认为 Phase 1 可直接部署——Teacher 依赖 **scandots 与 oracle waypoint**，必须完成 Phase 2。
- **局限（论文自述）：** 论文未设专门局限小节，讨论中仅指出下一步可扩展到移动操作；仅用单目前向深度、障碍为结构化 box / ramp 课程。
- **局限（工程，仓库 README）：** 栈绑定 **Isaac Gym Preview 3/4 + PyTorch 1.10** 等较旧环境；深度 **单目前向**、障碍以 **结构化 box course** 为主，野外 unstructured terrain 需自行扩展。后续 **人形 PHP**、**Hiking in the Wild** 等将跑酷推向更大机体与更长技能链。

## 参考来源

- [Extreme Parkour 论文摘录（arXiv:2309.14341）](../../sources/papers/extreme_parkour_arxiv_2309_14341.md)
- [extreme-parkour 代码仓库归档](../../sources/repos/extreme-parkour.md)
- [Extreme Parkour 项目页归档](../../sources/sites/extreme-parkour-github-io.md)

## 关联页面

- [Privileged Training（特权信息训练）](../concepts/privileged-training.md) — scandots / oracle heading → 深度 Student
- [DAgger](../methods/dagger.md) — Phase 2 行为克隆与在线聚合
- [legged_gym](./legged-gym.md) — 上游训练框架与工程范式
- [Locomotion](../tasks/locomotion.md) — 四足 RL 与跑酷任务地图
- [Sim2Real](../concepts/sim2real.md) — ROA / 域随机与部署叙事
- [DreamWaQ++](./dreamwaq-plus.md) — 四足感知 loco 姊妹路线
- [SWAP（对称等变世界模型跑酷）](./paper-swap-parkour.md) — RSSM + 硬等变约束；Apollo 纪录级 gap/box

## 推荐继续阅读

- [机器人论文阅读笔记：Extreme Parkour with Legged Robots](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Extreme_Parkour_with_Legged_Robots/Extreme_Parkour_with_Legged_Robots.html)
- Cheng et al., [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341)（ICRA 2024）
- Zhuang et al., *Robot Parkour Learning*（2023）— Unitree 生态 Teacher–Student 跑酷
- Kumar et al., *RMA: Rapid Motor Adaptation*（2021）— scandots→深度的 adaptation 先例
