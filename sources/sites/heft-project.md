# HEFT 项目页（heft.axell.top）

> 来源归档

- **标题：** HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum
- **类型：** site（项目页 + 演示视频 + 方法说明 + BibTeX）
- **URL：** <https://heft.axell.top/>
- **论文：** <https://arxiv.org/abs/2607.02332>
- **代码 / 检查点：** <https://github.com/Axellwppr/motion_tracking>（`main` 训练；`sim2real` 部署与已发布权重）
- **机构：** 清华大学、RobotEra、上海期智研究院；通讯 Jianyu Chen
- **作者：** Chenxin Liu*、Qingzhou Lu*、Guangxiao Yang、Xuanyang Shi、Chenghan Yang、Yanjiang Guo、Jianyu Chen（*Equal contribution）
- **硬件：** Unitree **G1**；全尺寸人形 **L7**（175 cm、65 kg）
- **入库日期：** 2026-07-04
- **一句话说明：** 从 **原始嘈杂 VR 参考** 跟踪人类意图，在 **G1 与 L7** 上实现 **重载全身遥操作**（双手负载至 **24 kg**）与高动态运动跟踪；核心为 **PMG（特权运动引导）** + **WPC（窗口化负载课程）** + **RMA 式师生适配器**。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Introduction | 重载 + 噪声 VR + 平衡约束并存；单策略部署于 L7 完成 locomotion、拾取、搬运、推物、负重深蹲 |
| Real-world Demos | 家务遥操作；基础 locomotion/拾放/深蹲/跪姿；**24 kg** 级双手负载；G1/L7 高动态跟踪 |
| Method | Reference-conditioned payload teleoperation；PMG；WPC（5 s 窗 + 专家可行负载上界）；师生控制器 |
| Evaluation | G1/L7 仿真；8 条噪声 VR、100 高动态 SEED、100 随机 SEED 持出集；对比 SONIC、TWIST2、消融 |
| BibTeX | `@misc{liu2026heft...}` arXiv:2607.02332 |

## 核心摘录（面向 wiki 编译）

### 1) 部署观测与训练目标

- 部署：$o_t^{\mathrm{dep}}=(S_{\mathrm{prop},t}, S_{\mathrm{raw},t})$，$a_t\sim\pi(\cdot\mid o_t^{\mathrm{dep}})$；低层 **PD** 跟踪关节位置命令。
- 训练：每 episode 采样参考运动与 **窗口化负载日程**；策略吃 **raw VR**，奖励对 **对齐后的 clean target** $S_{\mathrm{clean},t}^i$ 评估。
- 负载力施加于 **双手腕**；部署策略 **不直接观测负载状态**，鲁棒性靠训练动力学学习。

### 2) Privileged Motion Guidance（PMG）

1. 录制可部署 VR 轨迹，离线重建物理可行人体运动。
2. 将 raw / reconstructed 两路 **重定向** 为配对机器人参考。
3. Actor 吃 **raw**；reward / critic 用 **clean target**。

### 3) Windowed Payload Curriculum（WPC）

- 运动按 **5 s 窗口** 划分；用 clean-reference **专家 rollout** 估计每窗可行负载上界 $\bar F_{i,k}$。
- $F_{i,k}\sim\mathcal{U}(0,\bar F_{i,k}\,\mathrm{clip}(p/0.8,0,1))$；总负载在双手腕间随机分配，近重力方向施加并时域平滑。

### 4) 师生适配器（RMA 风格）

- Teacher：$z_t=E_p(S_{\mathrm{clean},t}, S_{\mathrm{priv},t})$。
- Student adapter：$\hat z_t=E_a(S_{\mathrm{prop},t}, S_{\mathrm{raw},t})$，$\mathcal{L}_{\mathrm{adapt}}=\|\hat z_t-z_t\|_2^2$。
- 部署仅保留 student + adapter；重建参考、窗口上界、负载状态、仿真特权信息全部移除。

### 5) 评测与基线（项目页公开）

| 对比项 | 要点 |
|--------|------|
| PMG 消融 | w/o noise（仅 mocap clean）；w/ noise（mocap + 通用噪声）；**PMG**（raw 输入 + clean 作训练引导） |
| G1 基线 | **SONIC**、**TWIST2** 官方 checkpoint |
| L7 基线 | **TWIST2** 在作者栈重训 |
| WPC 消融 | w/o expert（去掉专家引导窗上界）；**TWIST2+FC**（FALCON 式力课程迁入 TWIST2） |
| 负载扫描 | 双手总负载 0–30 kg 均匀分配 |

## 对 wiki 的映射

- 主实体：[HEFT（论文实体）](../../wiki/entities/paper-heft.md)
- 论文摘录：[heft_arxiv_2607_02332.md](../papers/heft_arxiv_2607_02332.md)
- 代码归档：[axellwppr_motion_tracking.md](../repos/axellwppr_motion_tracking.md)
- 任务交叉：[Teleoperation](../../wiki/tasks/teleoperation.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)
