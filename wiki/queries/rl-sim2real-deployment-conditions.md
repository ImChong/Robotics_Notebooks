---
type: query
tags: [sim2real, deployment, rl, locomotion, quadruped, observation, latency, actuator-model, domain-randomization, rma]
status: complete
updated: 2026-11-07
summary: "足式 RL 策略上真机时，先问仿真里究竟学会了什么：策略是在特定动作语义、观测来源、时序、身体与地面条件下才有效的控制方法，而非可原样搬运的模型文件。"
related:
  - ../concepts/sim2real.md
  - ./robot-policy-debug-playbook.md
  - ./sim2real-closed-loop-engineering.md
  - ./sim2real-checklist.md
  - ../concepts/domain-randomization.md
  - ../concepts/privileged-training.md
  - ../concepts/implicit-explicit-actuator-modeling.md
  - ../entities/paper-rma-rapid-motor-adaptation.md
  - ../entities/paper-tan-quadruped-agile-locomotion-sim2real.md
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
original:
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
  - ../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md
  - ../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/extreme_parkour_arxiv_2309_14341.md
sources:
  - ../../sources/blogs/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md
  - ../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md
---

> **Query 产物**：本页由以下问题触发：「强化学习真机部署时，机器人在仿真里究竟学会了什么？为什么维度对齐仍会翻车？」
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[RL 策略真机调试 Playbook](./robot-policy-debug-playbook.md)、[Sim2Real 闭环误差分层工程](./sim2real-closed-loop-engineering.md)、[RMA](../entities/paper-rma-rapid-motor-adaptation.md)；叙事骨架编译自 [深蓝具身智能 2026-09-18 公众号文](../../sources/blogs/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md)。

# RL 真机部署：策略成立的条件是什么？

## 一句话定义

仿真里学到的不是脱离条件的「走路本领」，而是在**特定动作解释、观测语义、时序、身体响应与地面反馈**下产生有效动作的控制方法；真机部署就是逐一核对并消化这些条件的变化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互优化长期回报的策略学习 |
| Sim2Real | Simulation to Real | 仿真训练策略迁移到真机 |
| PD | Proportional–Derivative | 底层高频关节跟踪；策略常输出其 setpoint |
| IMU | Inertial Measurement Unit | 真机姿态/角速度等本体感受主传感器 |
| DR | Domain Randomization | 训练时随机化仿真条件以提升鲁棒性 |
| RMA | Rapid Motor Adaptation | 从近期轨迹隐式估计环境并在线调整策略 |
| IPC | Inter-Process Communication | 进程间通信；共享内存/ROS 等数据路径 |

## 为什么重要

- **静默失败：** obs/action 向量长度一致、程序不报错，但关节顺序、默认姿态或缩放错一处，行为即完全不同。
- **误判迁移成功：** 短视频证明「能走几步」≠ 在目标载荷、地面与时长下具备任务能力；热保护、限流与多任务争用算力会在长时暴露。
- **工具用错层：** 接口/时序/映射问题应用部署对齐解决；DR/RMA 覆盖的是**已对齐接口后**的残余不确定性（见 [闭环工程](./sim2real-closed-loop-engineering.md)）。

## 流程总览：从「模型文件」到「控制链」

```mermaid
flowchart LR
  subgraph train [仿真侧 — 策略成立条件]
    A1[动作语义\n缩放·默认姿·关节序]
    A2[观测语义\n特权或估计状态]
    A3[时序\n策略周期·延迟假设]
    A4[身体模型\n理想或学习执行器]
    A5[地面/奖励\n接触与任务偏好]
  end
  subgraph deploy [真机侧 — 须同时成立]
    B1[动作链对齐]
    B2[传感器→同义状态]
    B3[端到端截止期]
    B4[响应轨迹匹配]
    B5[接触与长时工况]
  end
  train --> deploy
```

## 七类条件变化（部署排查主轴）

| 侧面 | 仿真常见假设 | 真机易错点 | 优先动作 |
|------|--------------|------------|----------|
| **1. 动作语义** | 输出为相对默认姿的偏移，经缩放后进 PD | 绝对角/顺序/符号/Kp 与训练不一致 | 对照训练部署脚本逐字段核对（如 Unitree-RL-GYM 路径） |
| **2. 观测来源** | 直接读 body vel、接触等 | IMU 安装系、滤波、估计延迟 | 禁止不可部署特权；对齐估计器输出语义 |
| **3. 感知→支撑** | 地形高度可踩 | 高程图可见 ≠ 可承重（植被、软面） | 区分「几何可见」与「力学可用」 |
| **4. 时序** | 瞬时或固定 dt | 策略低频 + 底层高频；队列延迟 | 测**端到端**延迟与 jitter，不只推理 ms |
| **5. 身体/执行器** | 理想力矩跟踪 | 摩擦、饱和、温升改变响应 | SysID / [执行器模型](../concepts/implicit-explicit-actuator-modeling.md)（ANYmal 学习执行器：[Actuator Network](../methods/actuator-network.md)） |
| **6. 地面** | 刚性平面 + μ | 形变、足垫磨损改变支撑反馈 | 勿把软地面简化成「换一个 μ」 |
| **7. 长时运行** | 无限 episode | 热限流、载荷、机载多模块 | 对照测试：地面/载荷/时长 + 版本绑定 |

## 七类条件的一手证据（2026-11-07 核对）

下表把上文七类条件逐项对到原论文，区分「原文明说」与「本页推断」：

| 条件 | 原文证据 | 出处 |
|------|----------|------|
| 动作语义 | 同样是「关节位置目标 + 固定增益 PD」，不同工作增益不同：RMA（A1）Kp=55、Kd=0.8；Margolis（Mini Cheetah）Kp=20、Kd=0.5；Hwangbo（ANYmal）kp=50 N·m/rad、kd=0.1。策略输出的数值含义随增益与缩放变化，不可跨平台照搬 | RMA §IV-A；Margolis §III-A；Hwangbo 方法 |
| 观测来源 | IMU 只给姿态的 2 个自由度，基座高度要靠腿运动学 + 卡尔曼估计，且腾空时不可用；Hwangbo 还发现把速度观测整体去掉会「完全训不出来」，而对观测速度加噪声（±0.5 rad/s 的关节速度噪声）使策略对不准确的速度更鲁棒 | Hwangbo 方法 |
| 时序 | RMA：策略 100 Hz、适应模块 10 Hz 异步，作者称该设计对算力有限的 A1「关键」；Digit：策略 50 Hz、PD 1 kHz；Extreme Parkour：深度延迟固定 0.08 s、本体 0.016 s；DreamWaQ++：本体 200 Hz / 外感知 10 Hz / 控制 50 Hz，训练里随机化 0–15 ms 延迟；Hwangbo：跌倒恢复策略训练于 20 Hz，部署到 100 Hz 性能一致（低速技能对控制率不敏感），而高动态行走需更高频率 | 各论文方法与部署小节 |
| 身体 / 执行器 | 理想执行器模型与手调解析模型（调了一周以上）训出的策略「一步都走不了」，肢体剧烈抖动，作者归因于延迟建不准；换成从真机数据学出的执行器网络后零样本部署成功。Digit 的闭链膝 / 踝机构在 Isaac Gym 里只能用高刚度虚拟弹簧近似，是额外的 sim2real 难点 | Hwangbo §结果；Radosavovic 方法 |
| 地面 | RMA 在训练中未见过的沙、泥、植被、乳胶床垫上成功；Margolis 发现盲策略的「粗糙地形鲁棒性」与「平地极速」存在取舍——训练地形越粗糙，平地可跟踪的指令范围越小 | RMA 图 1、图 3；Margolis 图 5 |
| 长时运行 | Hwangbo 的策略在三个月内未经修改地反复测试，期间机器人增加约 2 kg 配重并换了弹簧硬 3 倍的新驱动，策略仍稳定；热限流、电池压降与算力争用在这些论文中**没有**系统数据，本页相关描述为一手来源之外的工程经验（推测） | Hwangbo 讨论 |
| DR 与 RMA 的分工 | Margolis 表 IV：指令 6.0 m/s 时，真机有在线系统辨识 3.81 ± 0.09 m/s、无则 2.49 ± 0.07 m/s，仿真里分别 5.46 与 5.07 m/s——在线辨识显著缩小真机差距；RMA 论文的仿真基线中，仅靠域随机化的 Robust 策略表现不如带适应的 RMA，作者称 DR 以最优性换鲁棒（偏保守） | Margolis §IV-B；RMA §II、§V |

补充一条原文提醒（Hwangbo）：策略网络激活函数会显著影响真机表现——两种激活函数的策略在仿真里性能相近，在真机上可能差别很大；作者解释为无界激活（ReLU）遇到训练里没访问过的状态时动作幅值会很大，而 tanh 这类有界激活在受扰动时轨迹更温和。这类「仿真看不出的差异」是只靠仿真分数判断迁移成功的反例。

## 域随机化与 RMA：各管哪一段

- **DR** 应围绕**已校准基准**覆盖公差与难建模动态；扩大摩擦范围不会教会策略「软土下陷」，给 obs 加噪也不等于通信延迟。**映射/单位错误应直接修正**（见 [Domain Randomization](../concepts/domain-randomization.md)）。
- **[RMA](../entities/paper-rma-rapid-motor-adaptation.md)** 用近期状态–动作历史推断环境变化（载荷、μ 等），适合**运行中**工况漂移；**不能替代**动作/观测/时序对齐。

## 工程实践：最小验证序列

1. **支撑架 + 零指令：** 默认姿是否稳定；单关节正负阶跃方向是否正确。
2. **日志对齐：** 真机与仿真同指令下，打印 obs 各维语义与 action 解码后关节目标。
3. **延迟画像：** 采样→推理→下发全链路；对比训练假设 dt。
4. **短程 vs 长程：** 5 min 与 30+ min；记录驱动器温度/限流与算力占用。
5. **失效前抓数：** 失稳前目标–响应–时间戳错开方式，决定改模型、DR、观测还是时序（配合 [Playbook](./robot-policy-debug-playbook.md) 决策树）。

## 常见误区

1. **「权重一样就应该一样走」** — 忽略动作/观测解释链。
2. **「仿真高分即可上机」** — 奖励可能在鼓励仿真里可行、真机不可持续的力矩/步态。
3. **「DR 越大越安全」** — 可能学到过度保守；且无法修复明确 bug。
4. **「演示视频 = 部署完成」** — 未说明载荷、地面、时长与软件版本。

## 关联页面

- [Sim2Real](../concepts/sim2real.md) — 概念总览与 Domain Gap 方法索引
- [RL 策略真机调试 Playbook](./robot-policy-debug-playbook.md) — 训练/部署/硬件三类症状树
- [Sim2Real 闭环误差分层工程](./sim2real-closed-loop-engineering.md) — SysID→DR→适应闭环时序
- [Sim2Real Checklist](./sim2real-checklist.md) — 渐进式真机 SOP
- [RMA](../entities/paper-rma-rapid-motor-adaptation.md) — 在线适应模块
- [Actuator Network](../methods/actuator-network.md) — ANYmal 上学习执行器的一手方法页（Hwangbo 2019）
- [敏捷四足 Sim2Real（Tan et al., RSS 2018）](../entities/paper-tan-quadruped-agile-locomotion-sim2real.md) — Minitaur 上「解析执行器模型 + 动力学随机化」路线，与学习执行器对照
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- 一手论文通读归档（2026-11-07）：[Hwangbo 2019](../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md)、[RMA](../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md)、[Margolis 2022](../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md)、[Radosavovic 2023](../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md)、[DreamWaQ++](../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md)、[Extreme Parkour](../../sources/papers/extreme_parkour_arxiv_2309_14341.md)
- [深蓝具身智能：RL 真机部署条件（2026-09-18）](../../sources/blogs/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md)
- [深蓝具身智能：SysID→适应闭环（2026-07-28）](../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md)

## 推荐继续阅读

- [RMA: Rapid Motor Adaptation for Legged Robots（arXiv:2107.04034）](https://arxiv.org/abs/2107.04034) — 在线适应的一手原文
- [Learning Agile and Dynamic Motor Skills for Legged Robots（arXiv:1901.08652）](https://arxiv.org/abs/1901.08652) — 执行器建模与部署细节
- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/) — 产业侧部署坑汇总
- [Unitree RL Lab / RL GYM 官方部署文档](https://github.com/unitreerobotics/unitree_rl_gym) — 动作/观测转换参考实现
