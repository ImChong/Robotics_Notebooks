# Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control（arXiv:2610.09479）

> 来源归档（ingest）

- **标题：** Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control
- **arXiv：** <https://arxiv.org/abs/2610.09479>（v1，2026-10-07）；[HTML](https://arxiv.org/html/2610.09479v1)
- **项目 / 补充页：** [ResGAC](https://resgac.github.io/ResGAC-website/)；[实验详情](https://resgac.github.io/ResGAC-website/details.html)
- **作者 / 机构：** Joohwan Seo、Xiaofeng Guo、Jinkun Cao、Roberto Horowitz、Rocky Duan、Guanya Shi、Koushil Sreenath；UC Berkeley、Amazon Frontier AI & Robotics、CMU
- **平台：** Unitree G1（29 DoF：14 个手臂关节 + 15 个腿/腰关节）
- **代码状态（2026-11-10）：** 官方展示站点只公开静态网页，首页标注 `Code (released on acceptance)`；未提供可运行的训练、推理或控制器代码。
- **原文核对日期：** 2026-11-10
- **核对范围：** 通读 arXiv 官方 HTML v1 的摘要、I 引言、II-A/B 李群与问题定义（式 1–7）、III-A/B/C/D 几何导纳/残差强化学习/操作参考坐标系/训练（式 8–30、Tables I–II）、IV-A/B/C/D/E 五类实验小节（Tables III–IV、Peg-in-Hole、Mime-man）、V 结论与参考文献；另外核查项目页面首页与补充实现页的开放状态。
- **通读：** 已完成
- **一句话说明：** ResGAC 以几何导纳控制提供可解释的末端位姿跟踪底座，再用残差强化学习补偿动态误差并协调腿/腰，使 G1 在站立、移动基座和行走扰动下保持精确双手跟踪。

## 核心摘录

### 方法
论文针对人形机器人移动或承受全身扰动时的精确笛卡尔末端跟踪。单独的几何导纳控制（GAC）能将 SE(3) 位姿误差映射为手臂关节目标，但不足以处理步行、接触和未建模动力学；端到端 RL 则需要同时学精确几何控制与平衡。

ResGAC 将两者放进同一关节位置目标接口：GAC 产生名义手臂目标，RL 预测有界残差修正手臂目标，并直接输出其余 15 个腿/腰关节目标，学习步态和平衡协调。

### 参考坐标系
论文提出地面附着的 heading frame H0：保留平面位置和偏航，剔除 pelvis 的 roll、pitch、heave。相较将世界目标直接转入随 pelvis 移动的坐标系，H0 可减少躯干运动传入手部参考；它是坐标选择而非消除真实扰动的控制器。

### 实验结果
- **站立轨迹（G1 真机）：** setpoint MAE 为 6.5 mm / 1.37°；pick-and-place 为 6.6 mm / 1.67°。四种留出轨迹上优于 E2E RL、GAC + decoupled RL、SONIC；相对最佳基线位置误差降低约 54–67%，方向误差约 72%。
- **Peg-in-hole：** 25 mm 销插入 35 mm 孔，18/20（90%）；SONIC v1.1 为 10/20（50%）。失败包括目标注册误差与 OptiTrack 遮挡。
- **移动基座世界坐标跟踪：** ResGAC 8.8 mm / 2.69°，E2E RL 30 mm / 13.75°。
- **行走静态手保持：** H0 将 roll 传递比从 0.958 降到 0.790、z MAE 从 22.5 mm 降到 12.6 mm；pitch 传递没有改善（0.954 对 0.964）。

## 全文核验笔记（§III–V）

- **控制定义（§III-A2，式 13–18）：** GAC 利用 SE(3) 左不变几何误差、相对 body Jacobian、阻尼伪逆与零空间项计算双臂关节位置目标；作者明确说明此处 GAC **不使用实测外部 wrench**，不能将其描述成已验证的力传感器闭环接触控制。
- **联合动作（§III-B，式 19–22）：** 手臂动作 = GAC 名义目标 + 经缩放/裁剪的 RL 残差（s_r=0.75，c_r=1.0）；其余 15 个腿/腰关节位置由 RL 直接输出。actor 观测含关节/IMU、GAC 名义目标、任务误差、相对 twist、前两帧动作及步态相位；critic 在仿真训练期可额外看 pelvis 线速度。
- **参考系（§III-C，式 23–25）：** H0 只保留 pelvis 世界平面位置与偏航，移除 roll/pitch/heave 对末端命令参考的传播；实际足端冲击与全身动力学耦合仍存在。
- **训练（§III-D，Tables I–II）：** 25 项末端/步态/正则奖励，位置采用 L2²+L1 复合项、姿态采用 SO(3) trace 误差；课程 A–F 从站立固定目标递进到步行轨迹，逐步收紧跟踪误差；左右镜像 replay 增广及对称正则（式 30）。
- **部署（§IV-A）：** Isaac Sim / Holosoma / FastSAC，4096 环境，一张 L40S 完整六阶段约 16 小时；GAC 和策略均为 50 Hz，真机 Pinocchio 计算运动学，低层 PD 跟踪关节位置。
- **评测口径（§IV-B，Table III）：** 四种站立轨迹均留出；SONIC-Teleop-v1.1 用其公开模型的 heading 参考系，其余支持的基线使用 pelvis 参考系；setpoint 场景表格仅统计稳态样本。
- **坐标系消融（§IV-C，Table IV）：** H0 的 roll 传递比从 0.958 降至 0.790、z MAE 从 22.50 mm 降至 12.56 mm；pitch 传递比 0.964 对 0.954 基本不变。
- **操作实验（§IV-D）：** 站立插孔（直径 25 mm 销、35 mm 孔）ResGAC 18/20、SONIC-Teleop-v1.1 10/20；使用 OptiTrack 标记估计目标孔位，失败受到配准误差与标记遮挡影响。
- **世界系测试（§IV-E）：** 运动外环依赖 OptiTrack 对 pelvis 的定位；ResGAC 为 8.8 mm / 2.69°，E2E RL 为 30.0 mm / 13.75°，属于作者特定硬件/评测条件下报告值。

## 摘要标注修正

历史版本中“原文摘要”的英文块引用实为总结性改写，不是逐字原文。本次移除伪引文；如需原文措辞请直接查阅[官方 HTML v1](https://arxiv.org/html/2610.09479v1)。

## Wiki 映射

- 论文实体：wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md
- 任务综述：wiki/tasks/loco-manipulation.md
- 方法综述：wiki/methods/residual-policy-learning.md
