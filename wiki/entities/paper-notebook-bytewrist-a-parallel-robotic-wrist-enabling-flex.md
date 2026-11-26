---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-11-26
venue: curated
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_bytewrist-a-parallel-robotic-wrist-enabling-flex.md
summary: "ByteWrist（ByteDance Seed, arXiv:2509.18084v2）：嵌套三级电机驱动+弧形末端连杆+中心球铰支撑的紧凑型并联机械腕，在狭小手套箱抓取中耗时较 Kinova 串联腕减半（234 s vs 476 s），并支持 GR-3 完成 116 小时挂衣操作数据采集。"
---

# ByteWrist

**ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces**（ByteDance Seed，arXiv:2509.18084v2）提出一种面向狭小受限空间的紧凑型三自由度并联机械腕，目前已集成至 22-DoF 双臂移动机器人 ByteMini 中并用于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 硬件设计方向（分类：12_Hardware_Design）。

## 一句话定义

ByteWrist 是一种采用嵌套三级准直驱电机、弧形末端连杆与中心支撑球铰的 3-DoF 并联机械腕，兼顾狭小空间内的结构紧凑性、高刚度与拟人化 RPY 灵活运动能力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RPY | Roll-Pitch-Yaw | 末端并联平台绕 X/Y/Z 轴的横滚、俯仰与偏航三自由度姿态表示 |
| QDD | Quasi-Direct Drive | 准直驱执行器，ByteWrist 原型采用 RobStride Dynamics 的 QDD 电机 |
| SRS | Spherical-Revolute-Spherical | 球-转-球 7 自由度拟人机械臂构型（ByteMini 与对比基线 Kinova 均采用） |
| FK / IK | Forward Kinematics / Inverse Kinematics | 驱动连杆转角 $(\theta_1,\theta_2,\theta_3)$ 与末端 RPY 姿态间的正/逆运动学求解 |
| VLA | Vision-Language-Action | 视觉–语言–动作大模型，文中 ByteMini 用于支撑 GR-3 挂衣任务采集与部署 |

## 为什么重要

- 传统串联机械腕采用开链结构，存在误差累积、刚度低与前臂体积臃肿的问题，在橱柜、引擎舱或手套箱等受限空间内调整姿态极易发生前臂碰撞或被迫退出操作空间。
- 现有并联腕机构往往在「紧凑灵活」与「高刚度、大负载、简单传动」之间难以兼得；ByteWrist 通过嵌套驱动布局与中心球面支撑同时满足了高集成度与结构刚度，适合双臂拟人化灵巧操作。
- 论文提供完整几何参数表、正逆运动学与数值雅可比解法，并在 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与 [硬件设计分类页](../overview/paper-notebook-category-12-hardware-design.md) 中对应硬件本体层。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 作者与机构 | Jiawen Tian, Liqun Huang, Zhongren Cui, Jingchao Qiao, Jiafeng Xu, Xiao Ma, Zeyu Ren（ByteDance Seed） |
| 论文编号 / 日期 | arXiv:2509.18084v2（2025-09-24） |
| 项目主页 | <https://bytewrist.github.io/> |
| 深读状态 | 原文已核对（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/bytewrist-a-parallel-robotic-wrist-enabling-flex` |

## 核心机制

1. **嵌套三级并联驱动结构（Nested Three-Stage Drive）**：
   - 第 1 级电机固定于基座，驱动第 1 级驱动连杆并通过弧形末端连杆连接并联平台；第 2 级电机嵌装在第 1 级驱动连杆内部，第 3 级电机嵌装在第 2 级驱动连杆内部，显著压缩轴向与径向包络体积。
   - 驱动连杆与弧形末端连杆之间、弧形末端连杆与并联平台之间共 6 个转动副均指向并联平台中心。
   - 原型采用 RobStride Dynamics 的 QDD 电机，关键结构尺寸（单位 mm）：$R_1 = 27.35$、$R_2 = 30$、$h = 27.35$、$R_l = 25$、$l_1 = 5$、$l_2 = 13.68$，满足 $\theta_0 = \arctan(R_1/h)$、$R_l + l_2 = h/\cos\theta_0$ 与 $R_l + l_1 = R_2$。
2. **弧形末端连杆与中心支撑球铰（Arc-Shaped Linkages & Supporting Ball）**：
   - 90° 圆弧段加两段直段构成弧形末端连杆，优化力传递路径并扩大工作空间；平台中心安装支撑球并通过球铰与末端平台连接，在不牺牲姿态灵活性的前提下增强结构刚度。
3. **运动学与数值雅可比（Kinematics & Numerical Jacobian）**：
   - **正运动学（FK）**：给定三个驱动转角 $(\theta_1, \theta_2, \theta_3)$（初始位置为 $0^\circ, 240^\circ, 120^\circ$），利用并联平台上三点矢量两两夹角恒为 $2\pi/3$ 的三组非线性约束方程，以 **Newton-Raphson** 迭代求得弧形连杆转角 $(\phi_1, \phi_2, \phi_3)$，进而由平台法向量 $\mathbf{n}=(A,B,C)$ 算出旋转矩阵 ${}^1R_2 = R_Z(\alpha)R_Y(\beta)R_X(\gamma)$ 与 RPY 角。
   - **逆运动学（IK）**：给定目标 RPY 姿态，先构造 ${}^1R_2$ 得到末端连接点 $P_4, P_5, P_6$ 坐标，再反解 $(\phi_i, \theta_i)$。
   - **数值雅可比**：在 Creo 仿真中（步长 $\pi/3000$）对扰动步长 $\Delta\theta \in [10^{-7}, 10^{-1}]$ 进行误差对比，当 $\Delta\theta = 10^{-3}$ 时雅可比元素的最大误差与 RMSE 均达到最小。

### 方法流程图

```mermaid
flowchart LR
    CMD["目标姿态或驱动角指令"] --> MODE{"正逆运动学分支"}
    MODE -->|"IK (RPY 到驱动角)"| ROT["构造旋转矩阵 1R2 与平台点 P4-P6"]
    ROT --> SOLVE_IK["反解弧形连杆角 phi_i 与驱动角 theta_i"]
    MODE -->|"FK (驱动角到 RPY)"| NR["Newton-Raphson 求解 2pi/3 矢量约束"]
    NR --> RPY_OUT["计算平台法向量并提取 RPY 姿态"]
    SOLVE_IK --> QDD["嵌套三级 QDD 电机驱动连杆"]
    QDD --> ARC["三组弧形末端连杆 (6 个向心转动副)"]
    ARC --> BALL["中心支撑球铰与并联平台输出 RPY"]
    RPY_OUT --> JAC["数值雅可比 (最优步长 1e-3) 闭环控制"]
    JAC --> QDD
```

## 实验与评测

作者将 ByteWrist 集成至 22-DoF 双臂移动机器人 **ByteMini**（双 7-DoF SRS 臂、1-DoF 夹爪+RealSense D405、1-DoF 高刚度升降腰部、3-DoF 全向底盘、2-DoF 头部+RealSense D457、Dell T3280 主控、4.08 kWh 电池）完成三组实机实验：

1. **腕部工作空间与动态跟踪（Section 4.2）**：
   - 受结构约束，腕部俯仰 $\beta$ 与横滚 $\gamma$ 满足 $\beta^2 + \gamma^2 < 0.72$；在圆轨迹 $\beta^2 + \gamma^2 = 0.68,\ \alpha=0$ 下测试周期 $T = 4\text{ s}, 2\text{ s}, 1\text{ s}$。
   - 实测通信与计算延迟约 **0.06 s**；过零点跟踪误差分别为 **0.064 rad**（$T=4\text{ s}$）、**0.127 rad**（$T=2\text{ s}$）和 **0.247 rad**（$T=1\text{ s}$）。
2. **受限空间手套箱抓取对比（Section 4.3, Table 2）**：
   - 透明亚克力手套箱尺寸 $1000\text{ mm} \times 500\text{ mm} \times 600\text{ mm}$，前面板设两个直径 $200\text{ mm}$、中心距 $500\text{ mm}$ 的操作孔；保持腰部与底盘静止，使用 Quest VR 遥操作依次抓取 9 个物体（塑料娃娃、毛绒熊、浴花及 6 个甜甜圈玩具）。
   - **总耗时对比**：ByteMini（ByteWrist）总计 **234 s**（单件 17–49 s），对比同等 7-DoF SRS 构型但采用串联腕部的 Kinova Gen3 + Robotiq 2F-85 双臂平台总计 **476 s**（单件 23–75 s），耗时缩减约一半（快 2.03×）。
3. **双臂柔性衣物挂衣操作（Section 4.4）**：
   - 在基于 VLA 模型 **GR-3** 的胸前双臂协同挂衣任务（取衣架 → 挂左肩 → 换手 → 挂右肩 → 挂上横杆）中，ByteMini 累计完成 **116 小时** 遥操作数据采集并实现了全自动灵巧挂衣。

## 与其他工作对比

| 对比维度 | ByteWrist（ByteMini） | 传统串联腕（Kinova Gen3 双臂） | 传统并联腕（如线驱/球面机构） |
|----------|----------------------|-------------------------------|-----------------------------|
| 驱动布局 | 嵌套三级 QDD 电机+弧形连杆 | 串联 3-DoF 旋转关节（开链长前臂） | 分立底座驱动或柔索传动 |
| 狭小空间避障 | 腕部紧凑，孔内直接调姿无碰撞 | 需撤出孔外大角度转大臂、腕部相机易碰壁、前臂轻微碰撞 | 刚度与紧凑性往往难兼得 |
| 9 物体手套箱抓取总耗时 | **234 s** | **476 s** | 原文未同台实测 |
| 中心支撑刚度 | 中心球铰显著提升抗载刚度 | 悬臂开链刚度较低、误差累积 | 依赖支链几何或绳索预紧力 |

## 结论

**ByteWrist 通过「嵌套三级驱动 + 弧形末端连杆 + 中心支撑球铰」解决了并联腕在体积、刚度与灵活度上的三角权衡，使双臂在受限孔洞操作中的完成耗时相对同构型串联腕直接减半。**

1. **嵌套布局换取孔洞通过性** — 将第 2、3 级 QDD 电机逐级内嵌进前级连杆，消除外挂长前臂在狭小入口内的干涉，是手套箱任务提速 2×（234 s vs 476 s）的结构根因。
2. **数值雅可比步长需精确标定** — 正运动学依赖 Newton-Raphson 求解非线性约束，采用 $\Delta\theta = 10^{-3}$ 差分步长可使雅可比最大误差与 RMSE 同步最小化。
3. **高频动态控制需补偿系统延迟** — 实测 ~0.06 s 通信/计算延迟在 $T=4\text{ s}$ 慢速下影响较小（0.064 rad），但在 $T=1\text{ s}$ 高动态轨迹下误差放大至 0.247 rad，高速闭环需做延迟前馈或提频。
4. **已验证 VLA 数据采集可用性** — 支撑 GR-3 完成 116 小时挂衣数据采集与全自动部署，证明并联腕的拟人连续 RPY 运动适合作为双臂灵巧操作硬件底座。

## 常见误区或局限

1. **并非无限连续旋转**：由于并联连杆干涉约束，俯仰与横滚工作空间受限于 $\beta^2 + \gamma^2 < 0.72$，不适用于需要末端某一轴无限连续自转的任务。
2. **高速跟踪受限于通信延迟**：论文指出在 0.06 s 控制环延迟下，高速运动（$T=1\text{ s}$）时命令与状态曲线偏差达 0.247 rad，高动态任务必须计入延迟影响。
3. **工程成熟度待迭代**：作者在结论中明确指出，后续仍需优化结构参数以进一步扩大工作空间，并改进轻量化设计与内部电气走线可靠性。
4. **开源与硬件资料边界**：论文截至 arXiv v2 提供了项目页链接，但论文文本中未给出机械 CAD 或控制固件 GitHub 仓库；复现具体选型与走线需参考项目页更新。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_bytewrist-a-parallel-robotic-wrist-enabling-flex.md](../../sources/papers/humanoid_pnb_bytewrist-a-parallel-robotic-wrist-enabling-flex.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文原文：Jiawen Tian et al., *ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces*, arXiv:2509.18084v2 (<https://bytewrist.github.io/>)

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- [ByteWrist 项目主页](https://bytewrist.github.io/)
