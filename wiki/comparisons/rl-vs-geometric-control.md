---
type: comparison
tags: [rl, geometric-control, quadrotor, uav, control, comparison, engineering-selection, upenn]
status: complete
updated: 2026-08-26
related:
  - ../entities/paper-rl-vs-gc.md
  - ./mpc-vs-rl.md
  - ./wbc-vs-rl.md
  - ./model-based-vs-model-free.md
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ../concepts/domain-randomization.md
  - ../entities/isaac-lab.md
  - ../entities/gym-pybullet-drones.md
sources:
  - ../../sources/papers/leveling_playing_field_rl_vs_gc_arxiv_2506_17832.md
  - ../../sources/sites/pratikkunapuli-rl-vs-gc.md
  - ../../sources/repos/rl-vs-gc.md
summary: "RL vs 几何控制：先把目标函数、任务数据和前馈参考对称，再比方法。对称后 GC 更稳、RL 更快；敏捷与模型不确定选 RL，慢跟踪与可解释内环选 GC。"
---

# RL vs 几何控制：四旋翼跟踪怎么公平比、怎么选

空中轨迹跟踪里，学习控制器（以 PPO 为代表）和解析几何控制（\(SE(3)\) / DFBC）经常被写成「新方法全面更好」。UPenn GRASP 的 [RL vs GC](../entities/paper-rl-vs-gc.md)（RSS 2025）表明：**多数大差距来自实验协议，不是控制律类别的上限。**

## 一句话定义

> 先让两边拿到同一任务目标、同一数据分布和同一份未来参考，再谈 RL 还是 GC；对称之后 **没有总冠军**——GC 赢稳态与可解释内环，RL 赢瞬态、域随机化和未建模执行器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GC | Geometric Control | \(SE(3)\) 级联位置–姿态 PD + 微分平坦前馈 |
| RL | Reinforcement Learning | 仿真里用奖励学策略（本文语境多为 PPO） |
| DFBC | Differential-Flatness Based Control | 四旋翼平坦输出（位置+偏航）上的解析跟踪 |
| PPO | Proximal Policy Optimization | 空中/腿式并行仿真里最常见的 on-policy 算法 |
| DR | Domain Randomization | 把质量/惯量/推力等随机化；对 RL 更「免费」 |

## 为什么重要

- 学侧论文常用固件 PID 或悬停手调 GC 当基线，等于拿 **未为任务优化的解析器** 打 **为任务优化的网络**。
- 工程选型若信了被污染的结论，会在本来该用几何内环的慢跟踪上无谓上 GPU 策略，或在接球/大扰动上死守未调好的 PD。
- 同一套「三项对称」清单，也可以拿去审 [MPC vs RL](./mpc-vs-rl.md) / [WBC vs RL](./wbc-vs-rl.md) 里的跨类表格。

## 核心原理

### 必须对齐的三维

| 维度 | 问这句话 | 不对齐时会发生什么 |
|------|----------|--------------------|
| **目标** | 两边是否优化**同一**标量目标（奖励/代价）？ | GC 手调「够稳」、RL 直接最大化评测指标 → RL 看起来稳赢 |
| **数据** | 增益/权重是否在**评测任务分布**上调？ | 悬停调完的 GC 去跟 Lissajous，等于分布外 |
| **前馈** | 是否都能看到参考的未来（航点 horizon 或 \(\ddot p_d,\omega_d\)）？ | 砍掉平坦性前馈或用积分环顶替，GC 在敏捷段会系统性变差 |

### 对称之后还剩下什么真正的方法差

| 维度 | 几何控制 | 强化学习（PPO 类） |
|------|----------|-------------------|
| **稳态误差** | 结构上可收到 0 | 常有小偏置（奖励与容差退火的折中） |
| **瞬态** | 级联带宽受限，大误差收敛慢 | 无结构先验，可更猛地压初值误差 |
| **可调参数** | 约 8 个 PD 增益 | \(10^5\) 量级网络权重 |
| **模型不确定 / DR** | 参数少，吃不准质量/惯量时掉得快 | 同一分布上训过就较稳 |
| **电机延迟/饱和** | 刚体律对未建模执行器「既不崩也不学」 | 必须在含电机的仿真里训，否则会学 bang-bang |
| **可解释 / 部署** | 公式+增益；嵌飞控内环成熟 | 推理快但黑盒；本参考工作无真机数字 |
| **计算** | 每拍解析求值 | 离线并行 GPU 训练，在线一次前向 |

数字与表见 [论文实体页](../entities/paper-rl-vs-gc.md)（Lissajous Table IV、接球 Table V）。

## 工程实践

**什么时候选 GC**

- 参考可微、任务偏准、允许更长收敛时间（巡线、慢速末端保持）。
- 需要渐近误差到 0、要能向飞控组解释增益。
- 已经有可信动力学，且不打算做大范围 DR。

**什么时候选 RL**

- 时间窗紧（接球、穿越、从大扰动恢复）。
- 质量/惯量/推重比不确定，准备用 DR。
- 执行器有明显一阶延迟和饱和，愿意在匹配动力学的仿真里训。

**怎么比才算数（检查清单）**

1. 写出评测用的 \(r(t)\) 或 RMSE 定义，GC 用同一目标做自动调参（Optuna 即可），不要只手调。
2. 训练/调参轨迹族 = 测试轨迹族；悬停增益不得直接当 agile 基线。
3. 给 GC 前馈导数（或等价 horizon）；给 RL 同样长的未来航点。
4. 同时报 **奖励、RMSE、误差–时间曲线、下游任务成功率**；不要只丢一个 RMSE。
5. 若声称 sim2real：RL 与 GC 都要在同一 DR / 电机模型下重新优化，而不是只给 RL 做随机化。

开源试验台：[PratikKunapuli/rl-vs-gc](https://github.com/PratikKunapuli/rl-vs-gc)（Isaac Lab DirectRLEnv；见 [仓库归档](../../sources/repos/rl-vs-gc.md)）。轻量替代：[gym-pybullet-drones](../entities/gym-pybullet-drones.md)（默认不带这篇的对称 GC 协议）。

## 局限与风险

- 证据主体是 **四旋翼 + 固定臂仿真**；腿式 WBC、在线 MPC、视觉策略不能直接套数字。
- 「GC」在本文特指 Lee 风格 \(SE(3)\) 级联，不是 INDI / NMPC 的替身。
- 无真机；把仿真 RMSE 写进飞控验收会低估气动、延迟和状态估计误差。
- 仓库无 SPDX，产品化前先确认许可。

## 关联页面

- [RL vs GC 论文实体](../entities/paper-rl-vs-gc.md) — 表、时序图与复现入口
- [MPC vs RL](./mpc-vs-rl.md) — 在线优化 vs 策略；审表时同样检查目标/数据/信息是否对称
- [WBC vs RL](./wbc-vs-rl.md) — 人形融合架构；避免「手调 WBC vs 充分训练 RL」
- [Model-Based vs Model-Free](./model-based-vs-model-free.md) — RL 内部对照，不是本页的解析律
- [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [Isaac Lab](../entities/isaac-lab.md) / [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Sim2Real](../concepts/sim2real.md) / [Domain Randomization](../concepts/domain-randomization.md)

## 参考来源

- [leveling_playing_field_rl_vs_gc_arxiv_2506_17832.md](../../sources/papers/leveling_playing_field_rl_vs_gc_arxiv_2506_17832.md)
- [pratikkunapuli-rl-vs-gc.md](../../sources/sites/pratikkunapuli-rl-vs-gc.md)
- [rl-vs-gc.md](../../sources/repos/rl-vs-gc.md)
- Kunapuli et al., RSS 2025, [arXiv:2506.17832](https://arxiv.org/abs/2506.17832)

## 推荐继续阅读

- [项目页](https://pratikkunapuli.github.io/rl-vs-gc/) — 不对称表与接球视频
- Sun et al., *A comparative study of nonlinear MPC and DFBC for quadrotor agile flight* — **同类解析器之间**的公平对比，可与本页的跨类协议对照
