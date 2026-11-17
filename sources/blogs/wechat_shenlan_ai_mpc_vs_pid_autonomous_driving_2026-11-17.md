# 彻底搞懂 MPC｜PID 免费又好用，自动驾驶为什么还要上模型预测控制？

> 来源归档（blog / 深蓝 AI《算法详解》；依据用户上传的 13 页文章 PDF 通读整理，非逐字转载）

- **标题：** 彻底搞懂MPC｜PID 免费又好用，自动驾驶为什么还要上模型预测控制？
- **作者 / 发布方：** 深蓝学院 · 深蓝AI
- **类型：** blog / 控制方法教学
- **发表日期：** 2026-11-17 17:32（原文页面展示）
- **入库日期：** 2026-11-17
- **原文：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247606549&idx=1&sn=0292e34b9de6bf95696b2aaba6b02b8e
- **获取方式：** 用户上传的微信公众号文章 PDF 快照（13 页；页面时间 2026-11-17 20:51）；不提交 2.7 MB 二进制附件
- **核对范围：** 通读 PDF 第 1–12 页全部技术正文和第 11–12 页参考资料：PID 与预瞄/前馈的关系、MPC 预测模型、滚动时域、代价函数与软硬约束、Autoware MPC 横向与 PID 纵向、ACC、部署代价及局限；第 13 页为公众号推荐/目录。涉及的 Autoware、MathWorks 和 MIT 一手文档另行在线核对。
- **通读：** 已完成
- **一句话说明：** 对已有 MPC 方法页补充自动驾驶的工程化选型实例：未来参考轨迹 + 车辆模型 + 代价权衡 + 显式约束，滚动执行首个控制动作；与 PID 互补而非替代。

## 原文主线（按页码提炼）

1. **为什么不是只看当前误差（第 1–2 页）**：连续弯道里，即使当前横向误差为零，也需要考虑后续曲率变化；PID 可以配前馈、预瞄、增益调度和限幅，文章并未声称 PID 完全不能预判。
2. **Model + 优化（第 3–4 页）**：由车辆位姿/速度、转向输入和车辆运动学或动力学模型预测候选轨迹，结合规划参考轨迹求解控制序列；不是仅凭模型预测就构成 MPC。
3. **Receding Horizon（第 4–5 页）**：每周期计算未来 N 步，只下发第一步；下周期用新测状态和新参考重新求解，以减少模型偏差和外扰累积。
4. **代价与约束不同（第 6–7 页）**：跟踪误差、航向误差、转角及变化量可以作为加权代价；机械转角、转向速率和加速度限制适合显式约束；允许有限违反且惩罚的条件可设计软约束。
5. **系统分层（第 7–8 页）**：规划输出参考轨迹；MPC 既可以只做横向，也可以联动横纵向。Autoware 示例中横向 MPC 与纵向 PID 共存；ACC 纵向也可以用 MPC 兼顾速度、距离和加速度限制。
6. **工程成本（第 9–10 页）**：自行车模型在低附着、急转弯和延迟显著时有模型误差；时域长度、在线求解超时、约束冲突或不可行、权重调参都需处理。
7. **选型结论（第 10–11 页）**：简单弱耦合闭环可选 PID+前馈/补偿；涉及未来轨迹、多目标、执行器约束的场景才更值得 MPC；安全不能仅依赖优化结果。

## 一手资料交叉核验（2026-11-17）

- [Autoware 官方 MPC Lateral Controller](https://autowarefoundation.github.io/autoware_universe/main/control/autoware_mpc_lateral_controller/)：输入参考轨迹、里程计和当前转向，模型与 QP 求解器经 `setVehicleModel` / `setQPSolver` 配置，`calculateMPC` 给出横向控制。官方文档还描述滤波、模型/执行器限制。
- [Autoware 官方 PID Longitudinal Controller](https://autowarefoundation.github.io/autoware_universe/latest/control/autoware_pid_longitudinal_controller/)：纵向目标速度/加速度，含执行器延迟补偿与坡度补偿；不应把 PID 简化成“完全只看当前误差”。
- [MathWorks ACC System](https://www.mathworks.com/help/mpc/ref/adaptivecruisecontrolsystem.html)：MPC ACC 模块优化纵向加速度以跟踪速度并维持车距，同时施加速度/加速度/安全距离约束。这是官方示例，不代表所有量产 ACC 都采用 MPC。
- [MIT Underactuated Robotics：Trajectory Optimization / MPC](https://underactuated.csail.mit.edu/trajopt.html)：补充滚动时域的递归可行性与终端条件讨论；文章描述的“有约束”不等于天然安全或保证递归可行。

## 工程边界与开放状态

- 本文是教学文章，**不是独立 MPC 算法论文/项目**，没有声明专属可运行源码、车辆实验日志或新 benchmark；因此复用现有方法节点，不另建论文实体。
- 官方 Autoware 可浏览实现与文档，但本次仅核对上列官方模块文档，**未对其运行入口和调用链做源码级逐函数分析**，不将概念流程图标作“官方源码运行时序图”。
- “预测未来十步”是解释滚动时域的示例，不是所有 MPC 固定使用 N=10。
- 显式写入约束不等于自动避障或保证安全：模型不准、感知误差、求解不可行/超时必须有独立监测与降级策略。

## 对 wiki 的映射

- [MPC 方法页](../../wiki/methods/model-predictive-control.md) — 补充车载 PID/MPC 选型、滚动闭环、Autoware 和 ACC、落地验收清单。
- [PID 方法页](../../wiki/methods/pid-control.md) — 经典反馈与前馈、限幅的适用边界（本次未修改）。
- [Autoware](../../wiki/entities/autoware.md) — 作为实际模块化系统例子（本次未修改）。
- [经典模型控制纵深路线](../../roadmap/depth-classical-control.md) — Stage 3 已链接既有 MPC 方法页，无需重复挂载。
