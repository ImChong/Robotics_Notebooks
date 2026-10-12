# Feeling Through the Load 论文来源归档

## 元数据

- **题目：** Feeling Through the Load: Compliant Quadruped Locomotion under Payload Interactions
- **作者：** Shaunak A. Mehta, Mayank Mishra, Prajit KrisshnaKumar, Sebastian Scherer, Koichiro Niinuma
- **单位：** Fujitsu Research of America（匹兹堡）；卡内基梅隆大学 AirLab / Robotics Institute
- **预印本：** https://arxiv.org/abs/2610.08637
- **HTML：** https://arxiv.org/html/2610.08637v1
- **版本：** v1，2026-10-06，cs.RO，CC BY 4.0
- **项目页：** https://feelthroughload.github.io/ （[sources/sites/feelthroughload.md](../sites/feelthroughload.md)）
- **代码：** 项目页标 “Code (coming soon)”，GitHub 仓库 feelthroughload/feelthroughload.github.io 的 README 写 “Code and implementation details coming soon”，仅 docs 目录；截至 2026-10-12 未发布。

## 论文要解决的问题

人直接推拉四足背上的未固定载荷时，机器人要区分载荷自然滑动与有意引导力，既保持载荷平衡，又顺从地改变运动。方案把载荷交互当作“塑造运动的指令”，分开学习柔顺载运策略与只用机载量测的因果力估计器。

## 方法与训练信息

1. **平台：** Unitree Go2，躯干上有支撑，载荷为不固定刚体（高 0.20 m，底面 0.18×0.18 至 0.35×0.25 m）。
2. **策略：** PPO 非对称 actor-critic。actor 观测含阻抗指令、投影重力、关节位置/速度历史、上一动作、躯干 IMU、执行器力矩、3 维力通道及紧凑特权潜变量；critic 额外看未压缩特权状态；递归适应模块从观测恢复潜变量。历史长度未报告。
3. **交互模型：** 虚拟质量-弹簧-阻尼参考：M_v ẍʳ = K_p(xᵈ − xʳ) + K_d(ẋᵈ − ẋʳ) + F_L。页面称为阻抗控制式，不是严格导纳。
4. **奖励：** FACET 奖励加载荷余量、滑移、保持、等级项，系数 α_m、α_s、α_r、α_u 数值未给出。
5. **力估计器：** 递归编码器 G_φ，输入为去掉力通道的策略观测、足端法向载荷与接触状态；输出头为交互概率、单位方向、大小，并有因果滞回门；训练期另有质量头。网络类型、窗口长度、损失权重未完整给出。
6. **数据聚合与微调：** 先用盲策略与 oracle 力策略的 rollout 训练初始估计器，再做一轮 DAgger 式 estimator-in-the-loop 数据聚合并在并集上重训；actor 以 oracle 策略初始化，在估计力 rollout 下用 PPO 微调，并对冻结 oracle actor 加锚定 KL（权重 λ_A 未给）。
7. 仿真器、环境数与 PPO 超参原文未说明。

## 论文报告的结果

- **仿真设置：** 载荷 2/4/6 kg，力 6–30 N 作用 2 s、恢复 3 s；K_p ∈ {6,8,16,30}，K_d = 1.8√K_p，M_v = 4 kg；每条件 32 个随机实例，5 个种子。
- **Table I（ρ_cmd↑ / e_θ(°)↓ / 终止%↓）：** Oracle 0.865/1.77/19.13；Facet 0.252/8.99/52.91；Facet-L 0.702/7.27/21.57；Ours-Analytic 0.602/3.37/33.48；Ours 0.826/3.73/19.84。
- **Table II（估计器，学习 vs 解析）：** 幅值比 0.886 vs 0.445；方向误差 10.87° vs 14.24°；检测率 98.90% vs 86.84%；起始延迟 0.175 s vs 0.433 s。
- **Table III（避障，通过/联合成功/目标误差 m）：** Oracle 92.4%/92.4%/0.038；Facet-L 0.0%/0.0%/0.093；Ours-Analytic 34.8%/34.3%/0.055；Ours 77.1%/76.9%/0.068。
- **真机（Go2，载荷在原文写作 “44 kg”，疑似笔误，原文未核实）：** K_p = 8，M_v = 4 kg，标称 0.25 m/s 走 6 m，每方法 10 次，动捕仅用于测量。直线：Facet-L v_xy 0.259、路径偏差 0.025、目标误差 0.026、平均/峰值倾角 2.05/6.14°；Ours 0.252、0.045、0.042、1.70/7.40°。人引导：Facet-L v_xy 0.258、偏差 0.058、目标误差 0.089、倾角 2.07/8.38°；Ours 0.254、0.096、0.045、3.45/13.37°。Ours 中位倾角 3.44°，路径修正后平均侧向位移 0.345 ± 0.081 m。

## 复现与解释边界

- 仅平面载荷交互，假设载荷始终接触支撑；大力仍会导致载荷脱落。
- 未公开代码；估计器结构与多数超参缺失，难以复现。
- 真机载荷质量疑似印刷错误；每方法 10 次试验。
- 论文机构 Fujitsu Research of America 与同周 PAMORT 的 Fujitsu Limited 为不同署名。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-feeling-through-the-load-compliant-quadruped.md
- 周报汇总：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.08637>
- arXiv HTML 全文：<https://arxiv.org/html/2610.08637v1>
- 项目页：<https://feelthroughload.github.io/>
