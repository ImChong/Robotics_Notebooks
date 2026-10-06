# PerchRL: Vision-Based Agile Perching on Inclined Platforms under Rapid and Irregular Motion

> 来源归档（ingest · 论文 v3）

- **类型：** paper / aerial robotics / visual control / reinforcement learning
- **版本：** arXiv:2606.03441v3，2026-09-19
- **作者（v3）：** Zihong Lu, Zongzhuo Liu, Huaxu Li, Yitao Zeng, Jinqiang Cui, Jie Mei, Youmin Gong, U Kei Cheang, Boyu Zhou
- **单位：** Southern University of Science and Technology；Harbin Institute of Technology, Shenzhen；Peng Cheng Laboratory；Differential Robotics
- **论文：** [abs](https://arxiv.org/abs/2606.03441v3) · [HTML](https://arxiv.org/html/2606.03441v3) · [PDF](https://arxiv.org/pdf/2606.03441v3)
- **项目：** [STAR Group PerchRL](https://robotics-star.com/projects.html) — 独立归档 [robotics_star_perchrl.md](../sites/robotics_star_perchrl.md)
- **视频：** [1](https://www.bilibili.com/video/BV1t2j863ERi) · [2](https://www.bilibili.com/video/BV12LVm6yExG)
- **会议状态：** STAR Group 页面标注 Submitted to CoRL 2026；不据此写作已接收。
- **代码状态：** 论文原文称 source code will be released；本次核查时未发现官方代码链接。
- **入库日期：** 2026-10-06
- **一句话说明：** 两阶段策略先在动力学可行的随机 spline 轨迹上做状态预训练，再在含间歇 6-DoF 视觉检测的环境中微调；CTRV-EKF 维持预测连续性，可靠度衰减与主动感知奖励帮助策略在追击和重获视野间自适应。

## 摘要与版本确认

论文研究四旋翼在快速、非规则运动的倾斜移动平台上视觉栖停。主要困难来自有限 FOV、欠驱动飞行动力学与激烈机动：视觉反馈会间歇中断，平台运动轨迹又可能超出训练时的固定模式。

arXiv v3 元数据列出 9 位作者（相比早期版本增加 Yitao Zeng），修订时间为 2026-09-19。论文说代码将发布，并非已公开。官网将工作列为 CoRL 2026 在投。

## 方法摘要

1. **随机化轨迹。** 平台参考轨迹是周期均匀三次 B-spline。极坐标控制点经低通平滑，再做曲率检查/优化；扰动在时间上相关，以贴近差速车的平滑速度演化。
2. **状态预训练。** 策略用四旋翼状态、平台相对位置/法向/速度历史、上一动作等作为输入。TCN（3 个 1D 卷积层）编码历史，MLP actor 输出质量归一化 CTBR；asymmetric critic 可见未来 N=15 步平台状态。训练为 PPO。
3. **视觉微调。** “vision-based” 是模块化视觉系统：policy 输入来自上游检测器给出的间歇平台 6-DoF pose，不是 raw RGB 端到端策略。模拟针孔相机 FOV gate、各向异性/几何依赖检测噪声和系统偏差。
4. **缺测补偿。** CTRV 模型 EKF 在观测间持续预测，Mahalanobis gating 决定测量更新；输入加入预测状态和可靠度 `rho=exp(-lambda_rho*T_loss)`，显式提示估计因视觉丢失而老化。
5. **主动感知。** active reward 惩罚目标位置估计误差（按丢失时间增长）并鼓励相机光轴与目标方向对齐；同可见性感知输入一起训练，学习何时继续进场、何时优先恢复视觉。
6. **终端定义。** 接触后在平台坐标系检查切向距离、相对切向速度与法向姿态误差是否同时在阈值内；不是“碰到即成功”。

## 训练与评测设置

- PPO + Omnidrones；8192 并行仿真环境，100 Hz（0.01 s），500 步上限；报告机器 i7-14700KF / RTX 4080。
- domain randomization：质量 ±30%、惯量 ±10%、推重比 ±15%、相机 tilt ±5°、检测频率 20–60 Hz。
- 状态基线：Fast-Perching、InclineLander、去除 temporal augmentation 的 MLP；还评测 reciprocating-linear、racetrack、figure-eight 与随机 B-spline 运动。
- 视觉基线：同训练阶段的 LSTM；消融 temporal/state augmentation、可靠度输入和 active-perception reward。作者报告完整方法表现和收敛最好。输入增强本身贡献大；active reward 单独并不足以应对缺测。
- Crazyflie 2.1 验证状态策略可在静态 (90°) 斜面栖停；另一自制四旋翼用于移动平台视觉闭环。

## 关键真机结果（论文 Table II）

| 场景 | 速度 / 倾角 | 误差与成败 |
|---|---|---|
| Normal-I | 2.0 m/s / 70° | 平面误差 0.21 m；姿态对齐 12.12°；撞击法向/切向速度 −0.39/0.31 m/s |
| Hard-I | 0.7–2.7 m/s / 70° | 7/10 成功；平面误差 0.15±0.05 m；对齐 7.65±3.94°；感知丢失/接触/磁吸失败 1/1/1 |
| Hard-II | 1.5–2.2 m/s / 70° | 7/10 成功；平面误差 0.18±0.07 m；对齐 8.13±4.08°；失败模式 2/0/1 |

移动平台系统：610.2 g 自制机、推重比 2.26、PX4 + Jetson Orin NX，机载推理 100 Hz、<1.8 ms；1280×720 global-shutter 单目相机，FOV 82.96°×52.90°、约 15°向下倾角；Isaac ROS AprilTag 检测。平台是 Scout Mini 与 42×42 cm 可调倾角铁磁板，机底磁铁吸附。NOKOV 的 ground-truth 地面车状态只用于评测。

## 结论与局限

- 随机可行轨迹 + 历史编码针对固定轨迹过拟合；连续 EKF 状态 + 衰减可靠度解决预测器“继续给输入但策略不知道它已不可信”的缺口；主动感知 shaping 补充恢复目标视野的动机。
- perception 是独立的 detector，不是像素到动作；AprilTag 延迟、检测噪声和 FOV 建模是 sim-to-real 关键假设。
- CTRV-EKF 在突变加速度/转弯下可能偏离；可靠度代理按丢失时间单调衰减，不是经过校准的不确定度概率。
- 欠驱动动力学与刚性相机造成感知—控制耦合；作者计划研究主动云台。
- 硬场景 10 次中 7 次成功且有磁吸/接触/感知失败；样本量和平台配置限制外推。
- 论文代码尚待发布；复现需要重新实现环境/感知噪声/PPO配置，不能把 Omnidrones 的开源误写成 PerchRL 开源。

## wiki 映射

- 论文实体：[paper-perchrl-2606-03441.md](../../wiki/entities/paper-perchrl-2606-03441.md)
- 项目实体：[perchrl-project.md](../../wiki/entities/perchrl-project.md)
- 官方项目来源：[robotics_star_perchrl.md](../sites/robotics_star_perchrl.md)

## 原始入口

- arXiv v3 HTML / PDF（技术细节、训练设置、Table II、未来工作）
- STAR Group 项目与 Publications 页面（项目简述、CoRL 2026 投稿标注、作者行和演示视频）
