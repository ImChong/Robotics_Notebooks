---
type: entity
project_id: push-the-limit-humanoid-sprinting
tags: [paper, humanoid, locomotion, sprint, cornering, attention, amp, ppo, sim2real, noetix, tsinghua]
status: complete
updated: 2026-10-25
arxiv: "2610.14840"
related:
  - ../tasks/humanoid-locomotion.md
  - ../tasks/locomotion.md
  - ../methods/amp-reward.md
  - ../concepts/humanoid-policy-observation-inputs.md
  - ../concepts/humanoid-policy-reward-functions.md
  - ./paper-sprint-humanoid-athletic-sprints.md
sources:
  - ../../sources/papers/push-the-limit-humanoid-sprinting-arxiv-2610-14840.md
summary: "Push the Limit（arXiv:2610.14840，松延动力 / 清华大学，CoRL 2026）用因果多头注意力、航向方向奖励和技能条件 Multi-AMP 解决人形高速跑步的时序估计、转弯奖励与技能塌缩问题；论文报告仿真 6.5 m/s 跟踪和减重真机 100 m 平均 7.14 m/s。"
---

# Push the Limit：全尺寸人形高速冲刺与激进转弯

**Push the Limit**（*Learning High-Speed Sprinting and Aggressive Cornering on a Full-Scale Humanoid*，松延动力 / 清华大学，CoRL 2026，arXiv:2610.14840）研究如何让一台全尺寸双足人形在高速跑步时仍能稳定跟踪转向命令并保持动作先验约束。其三项设计分别处理本体感觉高频振荡、传统偏航奖励失效和多技能风格判别塌缩。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MHA | Multi-Head Attention | 在历史观测的时间维度提取不同动态关系的注意力编码器 |
| AMP | Adversarial Motion Priors | 通过判别器奖励让策略接近参考动作分布 |
| MAMP | Multi-Adversarial Motion Priors | 为不同动作技能设置独立 AMP 判别器 |
| PPO | Proximal Policy Optimization | 论文用于训练共享全身 locomotion 策略的强化学习算法 |
| DoF | Degrees of Freedom | 机器人可独立运动的关节自由度数 |
| Sim2Real | Simulation to Real | 将仿真训练策略迁移至物理机器人的流程 |

## 为什么重要

- 将高速跑步、激进转弯和动作风格监督放进同一人形策略中评估，重点处理超过约 4 m/s 后才明显出现的控制问题。
- 展示了一个具体的奖励设计信号：当步态本身造成躯干偏航振荡时，转向监督可以跟踪低频速度方向，而非要求瞬时偏航角速度追随命令。
- 与 [SPRINT 频谱先验](./paper-sprint-humanoid-athletic-sprints.md)形成有用对照：SPRINT 通过少量参考动作和频谱先验生成步态；本工作使用多技能 AMP 风格监督，并显式加入时序注意力与高速转向目标。

## 核心方法

### 单策略接口

论文将 locomotion 建模为 50 Hz 离散控制。单帧观测包含速度命令与技能 one-hot 标签，以及角速度、姿态、关节位置和速度、上一步动作等本体感觉；策略读取一段历史并输出 17 维目标关节角，低层 PD 控制器生成执行力矩。高速跑步技能由前向速度命令触发，原地转向由近零线速度与显著偏航命令触发。

### 三个针对性机制

1. **因果 MHA 历史编码器：** 以单层、双头因果 Transformer 汇总本体感觉历史，再与速度及技能命令融合。论文的 MLP 消融在约 6.0 m/s 附近饱和，而结构化时序编码方案可跟踪 6.5 m/s 指令；MHA 在激进转向实验中优于其 GRU、LSTM 与学习型估计器对照。
2. **航向方向奖励：** 高速时瞬时偏航角速度混合了真实转弯和交替落脚引起的躯干摆动。奖励改为监督基座速度方向与保持的目标航向对齐；低于阈值时仍回到偏航角速度跟踪。
3. **技能条件 Multi-AMP：** 站立、原地转向、行走、跑步分别有对应判别器，策略按当前命令技能取得风格奖励。将判别器合并的消融导致原地转向塌缩为近似静止，说明混合动作风格的判别分数未必能监督命令对应的技能。

## 实验与工程实践

| 项目 | 论文报告 |
|------|----------|
| 主平台 | 1.8 m、17-DoF 双足人形；10 个腿部关节、1 个腰部、6 个手臂关节 |
| 仿真 | Gazebo 运行与真机相同的 ONNX 策略及 ROS 控制器；策略控制频率 50 Hz |
| 仿真高速测试 | 6.5 m/s 阶跃速度指令跟踪；另测试冲刺中的恒定转弯命令 |
| 真机速度测试 | 标准跑道 100 m 计时：减重版主平台在 7.0 m/s 命令下跑完 14 s，平均 7.14 m/s；图示 6.5 m/s 命令部署未测地速 |
| 第二平台 | 1.36 m、40 kg、23-DoF 人形，100 m 用时 22 s，平均 4.55 m/s |
| 训练与迁移 | Isaac Lab + PPO；ONNX 零样本部署，无真机微调 |

原文报告的超过 7 m/s 是 100 m 计时测试的平均速度，测试用的是与主平台同执行器和运动学的减重迭代版。5.5 m/s 转弯图中的躯干内倾超过 25°；论文分别报告这些条件，不能将其合并理解为同一次 7.14 m/s 转弯测试。

## 与相邻方法对比

| 路线 | 运动风格先验 | 高速转向机制 | 论文中的真机速度结果 |
|------|--------------|--------------|----------------------|
| **Push the Limit** | 四技能对应的 Multi-AMP 判别器 | 高速用航向方向奖励，因果 MHA 编码历史 | 减重全尺寸平台 100 m 平均 7.14 m/s |
| [SPRINT](./paper-sprint-humanoid-athletic-sprints.md) | 5 条参考训练的频谱生成先验 + PPO 残差 | 速度连续外推，非本文的高速转弯机制 | Unitree G1 报告冲刺 6 m/s |
| [SD-AMP](./paper-unified-walk-run-recovery-sdamp.md) | locomotion 与 recovery 双判别器 | 重点是走跑与跌倒恢复统一，不是极速转弯 | G1 快速模式报告最高约 3 m/s |

这三条路线分别强调极速参考生成、技能条件风格判别和跌倒恢复统一；机器人形态、测试协议不同，速度数值不能直接作为同条件排名。

### 源码运行时序图

**不适用：** 截至入库日，arXiv 页面提供论文与演示视频，未列官方代码仓库或可运行实现。

## 局限与风险

- 高速时髋、膝俯仰关节接近力矩上限（论文报告约 99%，踝俯仰约 90%），最大速度下抗扰能力明显下降；仿真最大扰动下的恢复率低于中速条件。
- 训练只覆盖轻度地形变化；纯本体感觉方案尚未验证楼梯或非结构化地面。
- 跨形态验证限于两台运动学相近的人形，不能据此推断可直接迁移到结构差异更大的机器人。
- 公开材料中未见官方代码和数据链接，当前无法据此复现训练流程。

## 结论

这项工作表明，高速人形控制同时受时序状态估计、转向目标定义和技能风格监督影响，三者需要协同设计。

1. **读速度指标时注明协议：** 7.14 m/s 是特定减重真机平台跑道 100 m 计时平均值；仿真阶跃跟踪和转弯采用 6.5 m/s 指令。
2. **高速转向的监督目标要避开步态噪声：** 论文以低频速度方向替代高速段瞬时偏航角速度作为主要转向误差。
3. **共享策略不等于共享风格判别器：** 按命令选择技能对应的 AMP 判别器，避免原地转向在混合风格奖励中缺少有效学习信号。
4. **时序结构比堆大网络更关键：** MHA 参数量略少于 MLP 对照，增益来自处理历史观测的结构偏置；论文同时显示 GRU/LSTM 等也改善了直线高速跟踪。
5. **速度提升受执行器限制：** 真机最高速接近关节力矩上限，不能把策略结果理解为硬件仍有充足安全余量。

## 关联页面

- [Humanoid Locomotion](../tasks/humanoid-locomotion.md) — 人形移动任务与高动态技能索引
- [AMP & HumanX](../methods/amp-reward.md) — AMP 奖励与多技能风格判别
- [人形运控策略的观测输入](../concepts/humanoid-policy-observation-inputs.md) — 本体感觉历史与策略输入
- [SPRINT 人形竞技冲刺频谱先验](./paper-sprint-humanoid-athletic-sprints.md) — 频谱生成先验 + PPO 残差的冲刺对照

## 参考来源

- [论文来源归档](../../sources/papers/push-the-limit-humanoid-sprinting-arxiv-2610-14840.md)
- Huang et al., *Push the Limit: Learning High-Speed Sprinting and Aggressive Cornering on a Full-Scale Humanoid*, CoRL 2026, arXiv:2610.14840. <https://arxiv.org/html/2610.14840v1>

## 推荐继续阅读

- [arXiv HTML 全文](https://arxiv.org/html/2610.14840v1) — 方法、消融、真机 100 m 计时与局限。
- [演示视频](https://youtu.be/YWf62XkiApc)
