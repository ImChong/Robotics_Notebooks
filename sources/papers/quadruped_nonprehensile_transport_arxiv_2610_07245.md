# Robust Nonprehensile Object Transport 论文来源归档

## 元数据

- **题目：** Robust Nonprehensile Object Transport with Quadruped Robots
- **作者：** Ainoor Teimoorzadeh, Riccardo Pretto, Mario Selvaggio, Gokhan Alcan, Sami Haddadin
- **单位：** 慕尼黑工业大学 Munich Institute of Robotics & Machine Intelligence；坦佩雷大学；那不勒斯费德里科二世大学 PRISMA Lab；穆罕默德·本·扎耶德人工智能大学（MBZUAI）
- **预印本：** https://arxiv.org/abs/2610.07245
- **HTML：** https://arxiv.org/html/2610.07245v1
- **版本：** v1，2026-10-05，cs.RO，9 页 8 图
- **项目页：** https://quadtransport.github.io/icra/（匿名补充站，见 [sources/sites/quadtransport_icra.md](../sites/quadtransport_icra.md)）
- **代码：** 论文与项目页均未提供；截至 2026-10-12 未发布。

## 论文要解决的问题

四足机器人背上托盘运输不抓取的物体时，物体对惯性参数误差与机器人加速度十分敏感，容易滑落。论文不是只做更鲁棒的跟踪控制，而是先规划对不确定参数闭环敏感度最低的物体运动，再用耦合机器人与载荷动力学的凸 MPC 与全身 QP 去跟踪。

## 方法与训练信息

1. **不确定性感知轨迹优化：** 沿用 Giordano 等的闭环敏感度最小化；代价为状态敏感度矩阵 Π(t) 的 Frobenius 范数平方的时间积分。决策变量为平移与旋转的八次多项式系数；不确定参数为物体惯性参数与物体-平台摩擦系数。
2. **约束：** 首末端零速度与零加速度；用 Λ − r_Λ ⪰ 0 表达不滑动；物体质心高度 0.35–0.40 m；|v_b,x| ≤ 1.0 m/s、|v_b,y| ≤ 0.05 m/s、|v_b,z| ≤ 0.2 m/s。求解用 CasADi（MATLAB）+ IPOPT，初值取最小范数多项式 α₀ = M†d₀；Intel i9-10900K 上平均约 65 s。
3. **耦合凸 MPC：** 单刚体机器人模型 + 物体，24 维状态（机器人 12、物体 12）；输入为地面反力 f_g 与接触系数 Λ；物体与平台通过带松弛变量的加速度级软约束耦合。预测时域与 MPC 采样率原文未报告。
4. **全身 QP：** 决策变量为广义加速度（18）、关节力矩（12）与地面反力；约束为浮基动力学、支撑脚零接触加速度、摩擦锥、摆动脚零力与力矩限幅。
5. **接触模型：** 圆柱物体周边四点接触，摩擦锥以内接四棱锥近似（k=4，论文称偏保守）；增益 K_p = diag(250,15)，K_d = diag(15,0.2)。
6. **仿真：** MuJoCo，用动捕数据标定；圆柱物体名义质量 0.07 kg、半径 10 cm、高 20 cm，名义 μ = 0.52。机器人型号原文未命名。

## 论文报告的结果

- **轨迹：** 起点 (0,0,0.38) m，终点 x ∈ {2.0, 2.2, 2.4} m，时长 5 s；基线为固定姿态与直线轨迹（仿真），硬件仅直线基线。
- **仿真：** 30 次蒙特卡洛（惯性 ±10%）；摩擦质量研究为 5 个摩擦系数 × 3 个距离 × 30 个质量实现 = 450 次。
- **硬件：** 180 次试验 = 3 距离 × 2 轨迹类型 × 3 质量（名义与 ±10%）× 10 次重复。
- **主结论：** 优化物体运动相对固定姿态基线减少约 50% 滑动，相对直线基线减少约 30%，机器人质心跟踪误差也更低。其余数值只在图中给出，正文无表格数值（原文未报告具体数字）。
- **硬件现象：** 优化轨迹在中点附近滑动更明显，但最终位移更小。

## 复现与解释边界

- 硬件实验使用机器人默认控制器，并未运行所提全身控制器；耦合 MPC 与全身 QP 仅在仿真验证（来自原文局限）。
- 只测一个圆柱物体、±10% 参数变化；一阶敏感度近似与保守接触模型。
- 轨迹优化平均 65 s，属于离线规划，不能直接用于在线重规划。
- 未列代码；项目页为匿名补充站，视频无外链。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-quadruped-nonprehensile-object-transport.md
- 周报汇总：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07245>
- arXiv HTML 全文：<https://arxiv.org/html/2610.07245v1>
- 项目页：<https://quadtransport.github.io/icra/>
