# Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking（arXiv:2506.17832）

> 来源归档（ingest）

- **标题：** Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking
- **缩写：** RL vs GC
- **类型：** paper / quadrotor / geometric-control / reinforcement-learning / empirical-study / rss
- **arXiv：** <https://arxiv.org/abs/2506.17832>（Submitted 2025-06-21；PDF：<https://arxiv.org/pdf/2506.17832>）
- **会议：** Robotics: Science and Systems (RSS) 2025
- **项目页：** <https://pratikkunapuli.github.io/rl-vs-gc/> — 归档见 [`sources/sites/pratikkunapuli-rl-vs-gc.md`](../sites/pratikkunapuli-rl-vs-gc.md)
- **代码：** <https://github.com/PratikKunapuli/rl-vs-gc> — 归档见 [`sources/repos/rl-vs-gc.md`](../repos/rl-vs-gc.md)
- **作者：** Pratik Kunapuli、Jake Welde、Dinesh Jayaraman、Vijay Kumar
- **机构：** 宾夕法尼亚大学（UPenn）GRASP Lab
- **入库日期：** 2026-08-26
- **一句话说明：** 在四旋翼与固定臂空中机械臂轨迹跟踪上，指出先前 RL vs 几何控制（GC）对比常把目标函数、任务对齐数据与前馈参考三项不对称地只给 RL；对称后两类控制器差距远小于文献宣称，GC 稳态更好、RL 瞬态更好。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-26）：** <https://pratikkunapuli.github.io/rl-vs-gc/> 摘要写明开源几何控制与 RL 实现；页内 BibTeX 指向同一项目页。
- **仓库核查（2026-08-26）：** [PratikKunapuli/rl-vs-gc](https://github.com/PratikKunapuli/rl-vs-gc) 公开：Isaac Lab `DirectRLEnv`（悬停 / Lissajous / 接球）、`rl/train_rslrl.py`、`controllers/gc_tuning.py`（Optuna）、预训练 `rl/logs/rsl_rl/PaperModels/`。钉 **IsaacSim 4.2.0.2 / Isaac Lab 1.4.1 / Python 3.10**。GitHub **未声明 SPDX 许可证**（无 `LICENSE` 文件）。
- **结论：** **已开源**（训练 / 调参 / 评测 / 论文 checkpoint 齐全）；复用前需自行核对授权。评测全部在仿真，论文未给真机实验。

## 摘录 1：三条不对称（§I–§IV）

先前文献把「RL 优于 GC」写成共识，但跨类对比常继承各自社区惯例，导致 **RL 偏向**：

1. **任务目标（objective）** — RL 用任务奖励优化网络；GC 增益常手调到「够用」，且常在近悬停上调完再评跟踪。
2. **任务对齐数据（data）** — RL 在目标任务分布上采数百万步；GC 常用固件默认增益，未在同一 Lissajous 分布上调。
3. **前馈参考（feedforward）** — 微分平坦 GC 需要位置 4 阶、偏航 2 阶导数；许多基线用 PID 积分代替或把 \(\omega_d,\dot\omega_d\) 置零。RL 侧则常把未来航点 horizon 拼进观测。

本文协议：两边用 **同一奖励**（式 3）；GC 用 **Optuna 贝叶斯优化** 8 个 PD 增益；两边都给 **H=10** 未来位置+偏航；在同一 Isaac Lab 环境、同一初值族上评 1000 条 rollout。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-rl-vs-gc.md`](../../wiki/entities/paper-rl-vs-gc.md)；沉淀对比页 [`wiki/comparisons/rl-vs-geometric-control.md`](../../wiki/comparisons/rl-vs-geometric-control.md)。

## 摘录 2：控制器与任务（§III–§IV）

- **GC：** Lee et al. 的 \(SE(3)\) 几何跟踪（亦称 DFBC）：位置 PD 出期望加速度 → 由加速度与偏航构造 \(R_{des}\) → 姿态 PD + 前馈角速度/角加速度 → 合力推力 \(f_T\) 与体轴力矩 \(\boldsymbol{M}\)。观测 COM；固定臂末端跟踪时仍用 COM 律。
- **RL：** 体坐标系误差观测（位置/姿态/重力/速度/角速度误差，展平 \(\mathbb{R}^{21}\)）+ 未来航点；动作 \(\mathbb{R}^{4}\) 推力+力矩，clip 到 \([-1,1]\) 再缩放到平台限幅。3×256 MLP（ELU），约 27.6 万参数；RSL-RL PPO；位置容差 \(\delta_p\) 从 0.8 每 50M 步减半到 0.1。
- **任务：** 悬停（Lissajous 振幅为 0）与 Lissajous 跟踪；固定臂 0-DoF 空中机械臂是四旋翼的推广。仿真 100 Hz、控制 50 Hz；4096 并行环境、2 亿步，RTX A5000 约 30 分钟。

**对 wiki 的映射：** 实体页画协议与运行时序；交叉 [Isaac Lab](../../wiki/entities/isaac-lab.md)、[多旋翼栈](../../wiki/overview/multirotor-simulation-planning-control-stack.md)。

## 摘录 3：对称后的结论（§V–§VI / Table IV–VII）

| 设定 | 要点 |
|------|------|
| 任一不对称 | 单独打开都会拉大「到最大奖励的 gap」；手调悬停、无前馈、用 PID 代替前馈都会handicap GC |
| Lissajous best-in-class（1000 trial） | **四旋翼** RL 奖励 14.196 vs GC 13.447；**空中机械臂** GC 奖励 13.792 vs RL 13.621。GC 误差收敛到 0，RL 有稳态偏置；RL 瞬态更快，RMSE 对大初值扰动更敏感 |
| 接球（Table V，TTC=0.79 s） | RL-EE 0.65 / RL-COM 0.72 / GC 0.30；TTC≈2 s 时三类都接近 1。瞬态决定敏捷任务，不是「观察 COM 就注定输」 |
| 域随机化（质量/惯量/推重比 0–40%，评 20%） | RL 退化小；GC 因可调参数少、模型依赖，明显更差 |
| 一阶电机动力学 + 饱和 | 刚体上训的 RL-Simple 奖励掉到 2.94（bang-bang 不可行）；RL-Realistic 13.05。GC-Simple 与 GC-Realistic 接近（≈12.6–12.7） |
| 局限 | **纯仿真**；未逐篇复现先前论文的精确设定；未外推到其他本体 |

**对 wiki 的映射：** 对比页写「慢跟踪选 GC、高敏捷/不确定选 RL」；[Sim2Real](../../wiki/concepts/sim2real.md) / [Domain Randomization](../../wiki/concepts/domain-randomization.md) 补「DR 对解析 GC 不像对 RL 那样免费」。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-rl-vs-gc.md`**（流程总览 + 源码运行时序图 + 结论）。
- 新建 **`wiki/comparisons/rl-vs-geometric-control.md`**（三条不对称协议与选型）。
- 交叉：`mpc-vs-rl`、`wbc-vs-rl`、多旋翼栈、Isaac Lab、gym-pybullet-drones、Flightmare、sim2real、domain-randomization、reinforcement-learning。
