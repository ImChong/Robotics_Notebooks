# Joga: Learning Agile Humanoid Soccer Skills with Active Vision

> 来源归档（ingest）

- **标题：** Joga: Learning Agile Humanoid Soccer Skills with Active Vision
- **作者：** Avinash Narula、Khai Nguyen、Nolan Fey、Venki Parsuram、John Marangola、Martin Peticco、Pulkit Agrawal（前三位共同一作）
- **机构：** MIT Improbable AI Lab
- **类型：** paper（2026 preprint）
- **论文：** <https://joga.csail.mit.edu/static/joga.pdf>
- **项目页：** <https://joga.csail.mit.edu/>
- **代码：** 官方项目页公开了相机颈部硬件仓库；未列策略训练或部署代码。
- **入库日期：** 2026-10-20
- **通读：** 已完成
- **一句话说明：** 用主动视觉与数据驱动的感知、动力学对齐，让 Unitree G1 在真机上完成高速运球、接球、传球和射门。

## 核心摘录

1. **联合控制视线与动作。** Joga（Joint Optimization of Gaze and Action）使用带 RGB 相机的 2 自由度躯干安装式主动颈部；每个技能策略共同控制全身关节与颈部，训练采用任务驱动的仿真强化学习。
   - **对 wiki 的映射：** 主动视觉进入动作策略的控制边界；见 [Joga 实体页](../../wiki/entities/paper-joga-humanoid-soccer.md)。
2. **感知对齐模型（PAM）。** 研究者用 10 分钟配对的机载球检测与动捕数据，学习随机器人运动状态变化的检测偏差、噪声和漏检，并在策略训练时复现这些观测误差。
   - **对 wiki 的映射：** 面向真实传感器误差的定向仿真建模；见 [Joga 实体页](../../wiki/entities/paper-joga-humanoid-soccer.md)。
3. **全身执行器对齐与技能衔接。** 全身无监督执行器网络（WB-UAN）从 10 分钟真机运球数据学习全身关节位置指令修正，用修正后的仿真微调运球策略；再从运球访问到的机器人和球状态分布微调传球、射门策略，使技能可在行进中切换。
   - **对 wiki 的映射：** 将感知误差建模、执行器残差和技能状态分布连接成短数据 Sim2Real 流程；见 [Joga 实体页](../../wiki/entities/paper-joga-humanoid-soccer.md)。
4. **真机结果。** Unitree G1 持续运球速度为 1.71 ± 0.16 m/s，估计峰值 2.31 m/s；展示急转 135°、接住速度最高 5.2 m/s 的传球，并完成运球到传球/射门的技能切换。

## 复现边界

官方项目页说明，PAM 与 WB-UAN 是训练阶段使用的仿真侧模型；部署时运行技能策略、机载球检测和本体感知。项目页目前未提供策略代码、权重或完整训练配置；公开的 [Joga Neck 硬件仓库](../repos/joga-neck.md) 提供颈部结构、装配资料及 URDF / MuJoCo 模型，不等同于论文策略实现。

## 对 wiki 的映射

- [Joga 人形足球实体页](../../wiki/entities/paper-joga-humanoid-soccer.md)
- [Humanoid Soccer 任务页](../../wiki/tasks/humanoid-soccer.md)
- [官方项目页归档](../sites/joga-humanoid-soccer.md)
- [Joga Neck 硬件仓库归档](../repos/joga-neck.md)

## 来源

- [Joga 官方论文 PDF](https://joga.csail.mit.edu/static/joga.pdf)
- [Joga 官方项目页](https://joga.csail.mit.edu/)

## 原文通读核对（13 页，含附录 A–E）

- **第 1–2 页（摘要、引言、方法概览）：** 先训练数据采集策略，再采集真机检测、MoCap 与本体数据，拟合 PAM 和 WB-UAN；四项技能均用 PAM，运球另用 WB-UAN 微调，部署时两种对齐模型均不运行。
- **第 3 页（PAM）：** 球检测框以中心与对数尺寸表示；PAM 条件化预测四维框残差的均值、方差及可见概率，采用高斯负对数似然和可见性 BCE 训练，未检测到球时保持上次观测。
- **第 3–4 页（WB-UAN、DSC、主动视觉）：** 全身关节历史和命令输入网络，预测关节位置目标修正；通过接触一致的短时轨迹回放和 PPO 拟合，不需要力矩标签。DSC 从运球 rollout 的机器人/球状态初始化传球、射门。2-DoF 颈部与身体关节共同由任务奖励优化。
- **第 4–7 页（结果、消融）：** 方形路线真机成功数 PAM 4/4、DR 3/4、Linear 1/4；135° 转弯 DR 3/5、PAM 4/5、PAM+WB-UAN 5/5；高速运球成功轨迹平均球速由 1.35 提高到约 1.7 m/s；衔接真机运球→传球、射门分别由 4/5、3/5 提升至 5/5、5/5。
- **第 8 页（结论与参考文献）：** 数据驱动感知和执行器对齐优于仅扩大随机化的研究动机，结论限于论文所列实验。
- **第 9–10 页（附录 A–D）：** 主动颈部在宽角接球中漏接率 31.4%，锁颈为 60.6%；策略以 50 Hz 输出含颈部在内的 31 关节位置目标，物理仿真 200 Hz；actor 观测历史 3 步，critic 额外使用特权状态。真机由手柄操作者发出球速度方向及切换指令，并非完全自主比赛。
- **第 10–13 页（附录 D–E、表 IV–VIII）：** OAK-1 W 与 ChArUco/MoCap 校准；YOLO 使用仿真 on-policy 渲染微调，mAP50 从 0.6830 升至 0.9758；PPO 使用 16,384 环境、24 steps/env/iter、5 epochs、4 mini-batches；附录还列出奖励项、域随机化、课程进度和接触参数。

> 通读依据：用户提供的完整 13 页 Joga 论文 PDF；本归档为人工整理的逐节阅读记录，不代表官方已公开策略源码。
