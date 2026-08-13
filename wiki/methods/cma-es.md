---
type: method
tags: [optimization, black-box-optimization, evolution-strategy, sim2real, system-identification, actuator]
status: complete
updated: 2026-08-13
related:
  - ../concepts/sim2real.md
  - ../concepts/system-identification.md
  - ./actuator-network.md
  - ./joint-actuator-parameter-identification.md
  - ../entities/bam-better-actuator-models.md
  - ../entities/paper-pace-sim2real-legged-robots.md
  - ../entities/paper-notebook-sampling-based-system-identification-with-active.md
  - ../entities/paper-spot-rl-distributional-sim2real.md
sources:
  - ../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md
  - ../../sources/papers/pace_sim2real_arxiv_2509_06342.md
  - ../../sources/papers/spi_active_arxiv_2505_14266.md
  - ../../sources/papers/spot_rl_distributional_sim2real_arxiv_2504_17857.md
summary: "CMA-ES（Covariance Matrix Adaptation Evolution Strategy）是一种无梯度、黑箱的连续优化算法：维护一个多元高斯搜索分布，按采样个体的适应度自适应更新均值、步长与协方差矩阵。在本库里，它是 sim2real 中标定摩擦/执行器等难测仿真参数的主力工具。"
---

# CMA-ES（协方差矩阵自适应进化策略）

**CMA-ES**（*Covariance Matrix Adaptation Evolution Strategy*）是一种 **无梯度（derivative-free）、黑箱的连续优化算法**。它维护一个 **多元高斯搜索分布** $\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$，每代从中采样一批候选解、按适应度（目标值）排序，再据此 **自适应地更新均值 $\mathbf{m}$、步长 $\sigma$ 与协方差矩阵 $\mathbf{C}$**，使搜索分布逐步对齐目标函数的局部几何。它对 **非凸、噪声、不可导** 的中低维问题稳健，因而在机器人里常被用来标定「仿真里难以解析求导」的物理参数。

## 一句话定义

**用一个自适应的多元高斯采样分布做黑箱优化：采样—排序—更新均值/步长/协方差，无需目标函数梯度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CMA-ES | Covariance Matrix Adaptation Evolution Strategy | 本页方法：自适应协方差的进化策略 |
| ES | Evolution Strategy | 以采样-选择-更新为核心的黑箱优化族 |
| Sim2Real | Simulation to Real | 仿真到真机迁移，本页方法的主要应用场景 |
| MAE | Mean Absolute Error | 仿真-真机轨迹对齐常用的标定目标度量 |
| MMD | Maximum Mean Discrepancy | 分布差异度量，可作为 CMA-ES 的适应度信号 |
| PMSM | Permanent Magnet Synchronous Motor | 永磁同步电机，执行器物理参数的建模对象 |
| BAM | Better Actuator Models | 用 CMA-ES 辨识扩展摩擦参数的参考实现 |

## 为什么重要

- **仿真里有一堆「测不准」的参数：** 摩擦、阻尼、电机常数、接触恢复系数等既难手测、又对 sim2real 差距影响大。CMA-ES 把「让仿真轨迹贴近真机」写成黑箱目标，直接 **自动标定** 这些参数，省去逐个手调。
- **不需要可导仿真：** 物理仿真器（MuJoCo/Isaac）对参数的梯度往往不可得或不稳定；CMA-ES 只需前向仿真给出标量适应度，天然适配「仿真作黑箱」的标定回路。
- **中低维稳健：** 典型标定问题维度在几到几十维（如 12 关节四足约 49 维参数），正落在 CMA-ES 稳健的甜区；相较随机搜索/网格，它靠协方差自适应显著提高样本效率。
- **与其它建模路线互补：** 相比 [Actuator Network](./actuator-network.md)（数据驱动黑盒拟合执行器），CMA-ES 常用于 **解析/半解析参数** 的辨识，两者可组合（先解析标定再残差网络补偿）。

## 算法骨架

给定当前分布 $\mathcal{N}(\mathbf{m}_t, \sigma_t^2 \mathbf{C}_t)$，第 $t$ 代：

1. **采样：** 抽取 $\lambda$ 个候选 $\mathbf{x}_i = \mathbf{m}_t + \sigma_t\, \mathbf{y}_i,\ \mathbf{y}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_t)$。
2. **评估与排序：** 用目标 $f$（如仿真-真机轨迹 MAE）给每个个体打分并排序。
3. **均值更新：** 用前 $\mu$ 个精英的加权平均更新 $\mathbf{m}_{t+1}$。
4. **步长控制：** 依「进化路径」长度自适应缩放 $\sigma$（连续同向前进则放大步长，反之收缩）。
5. **协方差自适应：** 结合秩-1 与秩-$\mu$ 更新调整 $\mathbf{C}$，使采样分布沿有利下降方向拉伸。

其中协方差自适应是核心：它让搜索椭球逐步对齐目标函数的等值线形状，等价于在原空间上做了各向异性的自适应缩放。

## 主要技术路线

| 变体 | 针对的问题 | 要点 |
|------|------------|------|
| **标准 CMA-ES** | 中低维、连续、黑箱 | 秩-1 + 秩-$\mu$ 协方差更新的默认形态，机器人参数标定常用 |
| **sep-CMA-ES** | 维度偏高、算力受限 | 只维护对角协方差，$\mathcal{O}(n)$ 代替 $\mathcal{O}(n^2)$，牺牲各向异性换可扩展性 |
| **IPOP / BIPOP-CMA-ES** | 多模态、易陷局部最优 | 重启策略：收敛后以更大（或交替大小）种群重启，提升全局性 |
| **Active CMA-ES** | 需要更快收缩无效方向 | 在协方差更新里给劣质个体负权重，主动压缩不利方向 |
| **（对照）朴素 ES / 随机搜索** | 极低维或基线对照 | 无协方差自适应，样本效率明显低于 CMA-ES |

选路口径：机器人 sim2real 标定问题多落在「中低维 + 黑箱 + 含噪」，标准 CMA-ES 即可；维度上到上百再考虑 sep-CMA-ES 或降维参数化；目标函数明显多模态时用带重启的 IPOP/BIPOP。

## 在本库中的典型用法

| 场景 | CMA-ES 优化对象 | 适应度 | 代表页面 |
|------|------------------|--------|----------|
| **扩展摩擦标定** | 舵机 M1–M6 摩擦/电机参数 | 摆锤台架轨迹 MAE | [BAM](../entities/bam-better-actuator-models.md) |
| **足式关节动力学对齐** | ~49 维紧凑关节参数（$I_a,b,\tau_c$ 等） | 悬空 chirp 轨迹误差 | [PACE](../entities/paper-pace-sim2real-legged-robots.md) |
| **线性回归对照** | 有力矩时不必上 CMA-ES | Fourier + OLS | [关节执行器参数辨识](./joint-actuator-parameter-identification.md) / [FloBaRoID](../entities/flobaroid.md) |
| **腿足 base 惯量 + 主动探索** | mass / CoM / 惯量 / 电机模型；指令序列 | 轨迹误差 + FIM（D-最优） | [SPI-Active](../entities/paper-notebook-sampling-based-system-identification-with-active.md) |
| **难测仿真参数标定** | 摩擦、电机等难测项 | Wasserstein / MMD 分布差异 | [Spot RL 分布式 sim2real](../entities/paper-spot-rl-distributional-sim2real.md) |
| **物理参数手感对齐** | 球 restitution / friction | 落球与滚动试验对齐 | 人形足球技能类工作 |

## 选型与边界

- **适用：** 中低维（≲ 数十维）、黑箱、非凸、含噪的连续参数标定/整定；无可靠梯度时的默认之选。
- **不适用：** 高维（成百上千维）问题样本代价陡增，应转向可导仿真的梯度方法或参数化降维；对逐样本极贵的评估也需配合代理模型。
- **组合：** 常与 [系统辨识](../concepts/system-identification.md) 的实验设计、[Sim2Real](../concepts/sim2real.md) 的域随机化配合——先用 CMA-ES 把标称参数标准、再用随机化覆盖残余不确定性。

## 关联页面

- [Sim2Real](../concepts/sim2real.md) — CMA-ES 标定是缩小 sim2real 差距的一条主线
- [系统辨识](../concepts/system-identification.md) — CMA-ES 是其黑箱参数辨识的常用优化器
- [关节执行器参数辨识](./joint-actuator-parameter-identification.md) — 何时用 CMA-ES、何时用线性回归
- [Actuator Network](./actuator-network.md) — 数据驱动执行器建模，与解析参数标定互补
- [BAM（扩展摩擦模型）](../entities/bam-better-actuator-models.md) — 摆锤台架 + CMA-ES 辨识 M1–M6
- [PACE（足式 sim2real）](../entities/paper-pace-sim2real-legged-robots.md) — CMA-ES 拟合紧凑关节动力学
- [SPI-Active（采样式 SysID + 主动探索）](../entities/paper-notebook-sampling-based-system-identification-with-active.md) — CMA-ES 用于参数辨识与 FIM 指令优化
- [Spot RL 分布式 sim2real](../entities/paper-spot-rl-distributional-sim2real.md) — 以分布差异为适应度用 CMA-ES 标参

## 参考来源

- [BAM 论文归档（arXiv:2410.08650）](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) — 摆锤 CMA-ES 摩擦辨识
- [PACE 论文归档（arXiv:2509.06342）](../../sources/papers/pace_sim2real_arxiv_2509_06342.md) — CMA-ES 拟合关节动力学参数
- [SPI-Active 论文归档（arXiv:2505.14266）](../../sources/papers/spi_active_arxiv_2505_14266.md) — 采样式 SysID + 主动探索
- [Spot RL 分布式 sim2real 论文归档（arXiv:2504.17857）](../../sources/papers/spot_rl_distributional_sim2real_arxiv_2504_17857.md) — CMA-ES 标定难测参数
- CMA-ES 方法综述：Hansen, *The CMA Evolution Strategy: A Tutorial* <https://arxiv.org/abs/1604.00772>
