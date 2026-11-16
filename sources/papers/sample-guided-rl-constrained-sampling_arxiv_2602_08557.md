# Constrained Sampling to Guide Universal Manipulation RL（Sample-Guided RL，arXiv:2602.08557）

> 来源归档（paper）

- **标题：** Constrained Sampling to Guide Universal Manipulation RL（arXiv v1 首页标题；CoRL 2026 阅读清单所写标题为 *Combining Constrained Sampling and Reinforcement Learning for Manipulation*，推测为会议版标题）
- **类型：** paper
- **作者：** Marc Toussaint, Cornelius V. Braun, Eckart Cobo-Briesewitz, Sayantan Auddy, Armand Jordana, Justin Carpentier（TU Berlin / Robotics Institute Germany / LAAS-CNRS / Inria & DI-ENS）
- **arXiv：** <https://arxiv.org/abs/2602.08557>
- **PDF：** <https://arxiv.org/pdf/2602.08557v1>
- **代码：** 论文正文未给出代码仓库（只引用了 TD7 官方实现 https://github.com/sfujim/TD7，并提到随文视频）
- **入库日期：** 2026-11-12
- **一句话说明：** 用带碰撞 / 接触 / 力约束的模型求解器采样可行静态状态，再借轨迹或插值偏置 episode 的起点与目标，引导目标条件 RL（TD7）学习接触丰富的通用操作策略；仅仿真。
- **核对范围：** arXiv v1 PDF 全文（11 页：正文 8 页 + 致谢 + 参考文献），含摘要、引言、相关工作、问题形式化（式 1-9）、方法（IV-A 至 IV-D、算法 1）、评测（V-A 至 V-D、图 3-6）、结论；论文无附录 / 补充材料；随文视频与项目页未看。
- **通读：** 已完成

## 核心摘录

1. **问题形式化（III）：** Constrained Goal-conditioned MDP（CG-MDP）在标准目标条件 MDP 上加约束 g_c ≤ 0、h_c = 0，起点与目标分布 p0 = U(S_c × S_c)；约束含非碰撞 d_ij ≥ 0、接触点 / 法向力 / 库仑摩擦锥（式 6-8）与准静态牛顿-欧拉平衡（式 9），故起止状态均为静态可行状态。
2. **约束采样（IV-A）：** 随机选接触模式（支撑数 1-3）与盒内均匀样本 s̄，用增广拉格朗日解邻近问题（式 10）投影到可行集。
3. **轨迹数据（IV-B）：** 用 2 阶 B 样条参数化开环控制，CMA-ES 解式 12，黑箱仿真滚动，得到 D_u；可选 TD3+BC 式 BC 正则（IV-C，式 14，框架为 TD7）。
4. **五种起止采样（IV-D，式 16-21）：** traj / trajSched / 最近态插值 interp / interpSched / 直接采样 baseline；调度变量 α 从 0.1 线性升到 1。
5. **评测（V）：** 两个 MuJoCo 域，double sphere 与 panda sphere；TD7，总 2M 步、每 100k 步更新起止分布、Tsched = 1M；奖励为物体距目标 < 1 cm 的 0/1。
6. **结果（图 6 与文字）：** double sphere 上最佳随机起止成功率 83.1%（interpSched），去掉球上平衡类目标后 99.0%；panda sphere 上最佳 5.1%，baseline 0.001±0.001；BC 项影响小。

## 对 wiki 的映射

- [paper-sample-guided-rl-constrained-sampling](../../wiki/entities/paper-sample-guided-rl-constrained-sampling.md)
