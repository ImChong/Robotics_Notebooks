# Spot 高性能 RL 论文通读（四足 Sim2Real 视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/quadruped-robot.md`）

- **标题：** High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures
- **作者：** A.J. Miller, Fangzhou Yu, Michael Brauckmann, Farbod Farshidian（RAI Institute）
- **类型：** paper / quadruped / rl / sim2real
- **原文：** <https://arxiv.org/abs/2504.17857>（2026-11 通过 arXiv PDF v3 获取全文，pdftotext 转文本）
- **一句话说明：** 在 Spot 低层电机 API 上首次公开端到端 RL 策略部署；用 Wasserstein 距离与 MMD 度量仿真/真机数据分布差异，并以 CMA-ES 优化关节摩擦与力矩-速度限幅，实现 5.2 m/s 奔跑。
- **核对范围：** 通读摘要、§I–§V（相关工作、建模、相似度打分、CMA-ES、训练、部署、结果、结论）、全部图表说明与表 I；参考文献列表仅浏览，未逐条核对被引论文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **建模（§III-A）：** 5 ms 动作缓冲模拟 gRPC 通信延迟（实测延迟 2–6 ms 量级）；每腿 3 个电机（hipx、hipy、knee），膝带 remotized 丝杠，摩擦更高；髋力矩限幅为常数，膝力矩限幅取力矩-位置与力矩-速度限幅的较小者；观测噪声由状态估计器数据经 FFT 估计高斯 σ。
- **相似度打分（§III-B）：** 特征取关节位置、速度与策略动作；4 组脚本化指令序列各在真机采 5 次，仿真重复；用 Wasserstein 距离与 MMD 取均值，避免逐步轨迹比较的误差累积。
- **优化（§III-C）：** 8 个超参（髋/膝摩擦 + 6 参数力矩-速度曲线），CMA-ES 100 次迭代、种群 10；流程为默认参数训练 → 真机采数 → 优化 → 重训，一次迭代即够。表 I：髋摩擦 0.008 Nm、膝 0.180 Nm，力矩限 +97.00 / −108.79 Nm。
- **训练（§III-D）：** Isaac Lab + RSL RL 的 PPO，策略 50 Hz，PD 律 τ = kp(σa·a + qd − q) − kd·q̇，kp=60、kd=1.5、σa=0.2，200 Hz 施加；速度指令 Vx∈U(−2,5.5) m/s；奖励含速度跟踪、足端空中时间、离地高度、冲击/打滑惩罚及正则项；随机化地形、摩擦、基座质量、初始状态与推力；未随机化关节摩擦、增益、力矩限等。
- **部署（§III-E）：** Jetson Orin 上 ONNX 推理，经 gRPC 向低层 API 发送位置偏置、默认位置、零速度设定、kp/kd。
- **结果（§IV）：** 最高速度由 3.8 m/s 提升到 5.2 m/s（飞行相小跑），基线持续 3.7 m/s，默认控制器最快 1.6 m/s；湿滑肥皂水有机玻璃板上可恢复打滑；膝关节动作/力矩在仿真与真机更吻合；两策略在摆动腿回收阶段仍难产生负力矩，原因是建模未含电源功率分配限制。
- **结论（§V）：** 视觉感知、更好仿真建模与更激进行为留作后续。

## 对 wiki 的映射

- [`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)
- [`wiki/entities/paper-spot-rl-distributional-sim2real.md`](../../wiki/entities/paper-spot-rl-distributional-sim2real.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2504.17857>
