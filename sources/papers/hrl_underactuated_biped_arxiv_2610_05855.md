# 欠驱动双足分层 RL 避障 论文来源归档

## 元数据

- **题目：** Hierarchical Reinforcement Learning for Collision-Free Locomotion of an Underactuated Biped
- **作者：** Jagannath Prasad Sahoo, Saurabh Kumar, Surya Prakash S.K., Samiran Datta, Abhay Dwivedi, Amit Shukla
- **单位：** arXiv 页面未列；项目页署名为印度理工学院曼迪分校（IIT Mandi）
- **预印本：** https://arxiv.org/abs/2610.05855
- **HTML：** https://arxiv.org/html/2610.05855v1
- **版本：** v1，2026-10-05，cs.RO
- **项目页：** https://jaggu2606.github.io/biped-hrl-locomotion （归档见 [biped-hrl-locomotion.md](../sites/biped-hrl-locomotion.md)）
- **代码：** 截至 2026-10-12 未发布（项目页写 “Paper and Simulation Code will be fully released upon publication”，按钮为占位锚点）
- **入库日期：** 2026-10-12

## 论文要解决的问题

欠驱动双足机器人浮动基座无法被直接指令，每一次避障机动都会扰动平衡。论文要让同一个系统在保持动态步态的同时，根据局部障碍观测（含移动障碍）走向目标，而不是先规划路径再交给固定步态跟踪。

## 方法与训练信息

- **本体：** 仿真双足，8 个主动腿关节（每腿髋 yaw、髋 pitch、膝 pitch、踝 pitch），无髋/踝 roll；大腿与小腿约 99.45 mm，身高 542 mm，基座装 IMU。
- **仿真：** PyBullet，随机种子 42；单张 NVIDIA RTX A4000 训练。
- **高层（HL）策略：** 观测为基座位置与朝向、180° 内 36 条射线（截断 3.0 m）、最近 5 个动态障碍的位置与速度、机器人系下的局部目标；动作为体速度指令（vx, vy, ωyaw），tanh 压缩，每条指令保持 10 步（约 5 Hz）。射线和障碍状态是仿真特权量，用来替代激光雷达与目标跟踪器。
- **低层（LL）策略：** 观测为 8 个关节位置/速度、IMU 姿态与角速度、步态时钟 [sin ωt, cos ωt] 和当前指令；无障碍或目标信息。动作为 8 个关节目标位置（叠加在半蹲姿态上，PD 跟踪），50 Hz。
- **奖励：** LL 为速度跟踪、偏航率、步态项（接触时序一致、姿态正则、动作变化率、力矩惩罚）。HL 为路径进度（r_path）、姿态（r_posture，参考高度 0.45 m）、安全（r_safe，d_safe = 0.55 m，TTC 时域 1.5 s）、正则（卡住 0.7 s、过慢 < 0.02 m/s、常数时间惩罚）；shaping 总和截断到 −15；终止项为碰撞 −50（不终止回合）、跌倒 −50、到达 +30。HL 权重 w1–w15 为手调，原文未给数值。
- **算法：** 两层均用 SAC：折扣 0.99，学习率 1e-4，回放 1e6，batch 256，熵温度自动调节（初值 1.0），软更新 0.005，2×256 MLP，最大回合 5000 步（100 s）。
- **训练方式：** 双时间尺度联合训练，两层在同一回合内同时更新（LL 50 Hz，HL 5 Hz）；每回合随机化布局、障碍、起点与目标。总步数和回合数原文未报告，训练曲线为单种子。

## 论文报告的结果

每方法每条件 100 次试验（10 个未见布局 × 10 次，起终点相距 ≥ 6.0 m；动态障碍 0.10–0.30 m/s，最多 5 个）。基线共用冻结的同一 LL 步态，仅换指令生成器。

| 方法 | 静态 成功/碰撞 (%) | 动态 成功/碰撞 (%) |
|------|------|------|
| SAC+A* | 76.0 / 2.0 | 68.0 / 20.0 |
| SAC+RRT* | 78.0 / 22.0 | 68.0 / 32.0 |
| SAC+APF | 63.0 / 37.0 | 55.0 / 45.0 |
| HRL（本文） | 98.0 / 2.0 | 88.0 / 12.0 |

- 路径长度与 A* 参考相差 4% 以内（静态 7.92 m vs 7.64 m；时间 13.74 s vs 13.25 s）。
- 消融（多数仅见图）：去掉动态障碍速度/位置通道在静/动态下都严重退化；去掉 TTC 或距离惩罚影响相当；去掉姿态项失败转向跌倒；去掉 shaping 截断训练不稳（静态 90.0% 跌倒或超时）。

## 复现与解释边界

- 纯仿真，未做 sim-to-real；射线与障碍状态为特权信息，真实感知未验证。
- 单训练种子，种子方差未量化；基线假设已知静态地图，冻结步态与 HL 联合训练过，可能对基线指令跟踪不够忠实。
- 仅腿关节主动，不涉及上身或多接触避障。
- 无公开代码；HL 奖励权重原文未给，无法仅凭论文复现。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-hrl-underactuated-biped-collision-free.md
- 概念入口：wiki/methods/hierarchical-reinforcement-learning.md、wiki/methods/sac.md
- 任务入口：wiki/tasks/humanoid-locomotion.md、wiki/tasks/locomotion.md

## 一手来源

- arXiv 摘要与版本：<https://arxiv.org/abs/2610.05855>
- arXiv HTML 全文：<https://arxiv.org/html/2610.05855v1>
- 项目页：<https://jaggu2606.github.io/biped-hrl-locomotion>
