# Leveling the Playing Field 论文通读（RL vs 几何控制 · 多旋翼栈选型视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/multirotor-simulation-planning-control-stack.md`）

- **标题：** Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking
- **作者：** Pratik Kunapuli, Jake Welde, Dinesh Jayaraman, Vijay Kumar（UPenn）
- **类型：** paper / quadrotor / rl-vs-geometric-control
- **原文：** <https://arxiv.org/abs/2506.17832>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext 提取）
- **一句话说明：** 指出 RL 与几何控制（GC）比较中的三类不对称（目标函数、任务数据、前馈信息），在对称协议下两者差距远小于既有文献，GC 稳态更优、RL 瞬态更优。
- **核对范围：** 通读摘要、§I–§VII（相关工作、GC 与 RL 定义、方法学三类不对称、实验 §V-A 至 §V-F、讨论与局限、结论）及附录 A–E（任务随机化范围、GC 实现、RL 网络与 PPO 超参、电机一阶动力学、完整结果表 IX）；参考文献仅浏览条目，未逐篇核对被引文献。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三类不对称（§I、§IV、表 I）：** (1) 目标函数：RL 优化任务奖励而 GC 常手调；(2) 数据：RL 在任务对应轨迹上训练，GC 增益常在悬停下调好后用于所有轨迹；(3) 前馈：GC 本应使用参考轨迹最高 4 阶位置导数与 2 阶偏航导数，却常被 PID 积分项替代或置零。
- **GC（§III-A、附录 B）：** 位置 PD 外环算期望加速度（式 4，含前馈 p̈d），再得期望姿态，姿态 PD 内环输出总推力与力矩；也称 SE(3) 控制或微分平坦控制。
- **RL（§III-B、附录 C）：** 观测为机体系位置/姿态/速度/角速度误差与重力向量（21 维），动作 4 维（推力+力矩，归一化后缩放）；3 层 256 单元 ELU 网络，PPO（RSL-RL），4096 并行环境，Isaac Lab 仿真，100 Hz 动力学 / 50 Hz 控制，200M 步约 30 分钟。
- **公平协议（§IV）：** 同一奖励（式 3）既训练 RL 又用 Optuna 贝叶斯优化整定 GC 的 8 个增益；两者都在同一 Lissajous 任务分布上优化；两者都给 H=10 的未来参考窗口；位置容差 δp 退火（0.8→0.1）是 RL 兼顾敏捷与零稳态误差的关键。
- **消融（§V-B、图 2）：** 手调→优化、悬停数据→Lissajous 数据、PID→前馈，对 GC 与 RL 都有收益，Lissajous 跟踪上差异更大；PID 替代前馈甚至比无前馈更差。
- **最佳对最佳（§V-C、表 IV）：** 四旋翼平均奖励 RL 14.196 vs GC 13.447；位置 RMSE 0.119 vs 0.158 m；但 GC 稳态误差收敛到 0，RL 留有轻微偏移。作者结论：GC 在奖励目标上胜出，RL 仅因初始瞬态更好而 RMSE 略低，与既有文献结论相反。
- **敏捷任务（§V-D、表 V）：** 接球任务中 RL-EE 成功率明显高于 GC（如 Time-to-Catch 0.79 s：0.65 vs 0.30）；RL-COM 也优于 GC，说明限制来自 GC 分层结构而非仅观测量。
- **域随机化（§V-E、表 VI）：** 质量/惯量/推重比随机化 20% 与 40% 下 RL 退化较小，GC 退化明显；作者认为 RL 更适合需用仿真覆盖不确定性的场景。
- **真实动力学（§V-F、表 VII）：** 仅用刚体简化动力学训练的 RL 在含电机延迟与饱和的环境中失效（奖励 2.941），在真实动力学下训练的 RL 达 13.053；GC 在两者间差异小。
- **局限（§VI）：** 全部在仿真中，未上真机；未复现前人完整实验；结论是否推广到其他构型/任务留待未来。

## 对 wiki 的映射

- [`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)
- [`wiki/entities/paper-rl-vs-gc.md`](../../wiki/entities/paper-rl-vs-gc.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2506.17832>
