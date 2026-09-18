---
type: query
tags: [sim2real, deployment, rl, locomotion, quadruped, observation, latency, actuator-model, domain-randomization, rma]
status: complete
updated: 2026-09-18
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
  - ../entities/paper-quadruped-agile-sim2real-rss2018.md
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
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
| **5. 身体/执行器** | 理想力矩跟踪 | 摩擦、饱和、温升改变响应 | SysID / [执行器模型](../concepts/implicit-explicit-actuator-modeling.md)（ANYmal 学习执行器先例） |
| **6. 地面** | 刚性平面 + μ | 形变、足垫磨损改变支撑反馈 | 勿把软地面简化成「换一个 μ」 |
| **7. 长时运行** | 无限 episode | 热限流、载荷、机载多模块 | 对照测试：地面/载荷/时长 + 版本绑定 |

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
- [ANYmal 敏捷 Sim2Real（RSS 2018）](../entities/paper-quadruped-agile-sim2real-rss2018.md) — 学习执行器模型先例
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [深蓝具身智能：RL 真机部署条件（2026-09-18）](../../sources/blogs/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md)
- [深蓝具身智能：SysID→适应闭环（2026-07-28）](../../sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md)

## 推荐继续阅读

- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/) — 产业侧部署坑汇总
- [Unitree RL Lab / RL GYM 官方部署文档](https://github.com/unitreerobotics/unitree_rl_gym) — 动作/观测转换参考实现
