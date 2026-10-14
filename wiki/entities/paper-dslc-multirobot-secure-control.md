---
type: entity
project_id: dslc-multirobot-secure-control
tags:
  - paper
  - multi-robot
  - secure-control
  - distributed-control
  - dmpc
  - nudt
  - hunan
  - university-of-manchester
status: complete
updated: 2026-10-14
arxiv: "2609.06896"
related:
  - ../methods/model-predictive-control.md
  - ../methods/reinforcement-learning.md
  - ../concepts/distributed-systems-basics.md
sources:
  - ../../sources/papers/dslc_multirobot_secure_control_arxiv_2609_06896.md
summary: "DSLC 将多机器人分布式 MPC 建模为隐蔽执行器攻击下的零和微分博弈，并用 attacker–actor–critic 学习可扩展的闭环防御策略；在 1000 机器人仿真和轮式机器人实验中验证。"
---

# DSLC：隐蔽执行器攻击下的多机器人安全学习控制

**Distributed Secure Learning Control（DSLC）** 将分布式模型预测控制与对抗式策略学习结合，在多机器人系统中同时优化攻击者和防御者策略，以应对隐蔽执行器攻击。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DSLC | Distributed Secure Learning Control | 本文提出的分布式安全学习控制框架 |
| MRS | Multi-Robot System | 多台机器人通过局部通信协同完成任务的系统 |
| DMPC | Distributed Model Predictive Control | 各机器人基于邻居信息滚动生成局部控制策略 |
| RL | Reinforcement Learning | 用于学习攻击与防御策略的强化学习方法 |
| IAE | Integral of Absolute Error | 实验中用于衡量控制误差的积分指标 |

## 为什么重要

分布式控制依赖机器人之间的邻居信息交换，因此被攻陷的节点可能通过控制输入扰乱整体协同行为。隐蔽执行器攻击会伪造看似正常的控制信号，传统检测或稳定性导向的防御难以同时保证系统性能。本文把安全防御表述为对抗性控制问题，并针对编队与 containment 等任务提供统一框架。

## 核心机制

### 方法栈

在每个滚动预测区间内，攻击者与防御者构成有限时域零和微分博弈。攻击者选择使协同性能变差的执行器输入，防御者生成抵抗攻击的控制输入。每个机器人仅使用本地状态及邻居信息，形成分布式策略。

作者提出 attacker–actor–critic 结构，沿预测时域逐步分解策略学习子问题，在线学习攻击与防御策略，并以解析的闭环策略替代传统 DMPC 每步对完整开环控制序列进行数值优化。该框架统一用于一般编队、仿射编队和 containment 控制。

### 流程总览

```mermaid
flowchart LR
    A["机器人局部状态与邻居信息"] --> B["预测区间内的零和微分博弈"]
    B --> C["分布式 attacker–actor–critic"]
    C --> D["闭环攻击策略与防御策略"]
    D --> E["执行当前控制并滚动至下一预测区间"]
    E --> A
```

## 工程实践

- **控制对象：** 连续时间非线性移动机器人；每个机器人基于本地状态与通信邻居决策。
- **任务覆盖：** 一般编队、仿射编队与 containment。
- **实验路径：** 仿真研究覆盖 2、4、6 机器人对比及最多 1000 机器人规模验证；实机实验在多台轮式机器人上完成。
- **规模迁移：** 论文报告由较小规模训练得到的防御策略可推广到规模不同的机器人群体及不同攻击概率条件，减少重新求解控制器的负担。
- **代码复现：** 截至 2026-10-14，未发现作者公开的项目页或可运行代码仓库。源码运行时序图不适用；复现需依据论文方法与补充材料自行实现。

## 实验与评测

论文比较 DSLC、DRHRL 与 DRMPC 等基线，使用 IAE、ITAE、任务完成时间和单步计算时间评估控制效果与实时性。作者报告：

- DSLC 在多机器人编队、仿射编队与 containment 实验中，相较对照方法取得更低的控制误差指标。
- 在 100 台机器人仿真中，DSLC 的平均计算时间约为 10.1–10.2 ms；表中 DRMPC 未能在对应条件下给出可比结果。
- 轮式机器人实测结果支持其在多种任务和攻击强度下运行，但不能直接推断到空中机器人或其他动力学系统。

读数值时应同时核对机器人规模、攻击概率、任务定义及计算平台；大规模仿真结果不等同于同等规模的真实系统验证。

## 与其他工作对比

| 路线 | 在线控制计算 | 安全/攻击处理 | 本文中的定位 |
|------|--------------|----------------|--------------|
| 数值优化式 DMPC（DRMPC） | 每个预测区间求解优化问题 | 通常需额外建模攻击或鲁棒约束 | 大规模下计算负担可能成为瓶颈 |
| 学习式 DMPC（DRHRL） | 用策略学习加速策略生成 | 本文将其作为性能对照 | 不具备本文的攻击者—防御者联合博弈结构 |
| DSLC | 在线学习解析闭环攻击与防御策略 | 直接针对隐蔽执行器攻击进行零和优化 | 兼顾滚动预测控制、对抗防御与多任务统一建模 |

## 结论

**DSLC 的主要价值是把多机器人安全控制中的性能优化显式建模为分布式对抗学习问题，并用闭环策略降低在线求解负担。**

1. **对抗目标要纳入控制器设计。** 它同时学习攻击与防御策略，而非只把攻击当成额外噪声。
2. **规模结果有条件。** 1000 台机器人是仿真扩展结论；论文的实机证据来自多台轮式机器人。
3. **策略迁移依赖假设。** 当前框架要求已知攻击概率，并假设静态通信拓扑、无通信时延。
4. **安全性结论需按证明条件理解。** 闭环稳定性与性能保证建立在论文给定的系统模型和假设上，不应外推为对任意攻击或未知动力学都鲁棒。
5. **复现仍有门槛。** 当前没有发现公开可运行实现，性能数字暂不能通过独立代码复现核验。

## 局限与风险

- 静态通信拓扑和无通信时延限制了动态组网、多跳不稳定链路等场景的直接适用性。
- 训练和策略设计假设攻击概率已知；真实部署中该量可能难以准确估计。
- 实机验证范围局限于轮式地面机器人；无人机等其他平台尚属未来扩展方向。
- 未发现作者提供的公开代码入口，需将论文实验结果与独立复现结果区分看待。

## 关联页面

- [model-predictive-control](../methods/model-predictive-control.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [distributed-systems-basics](../concepts/distributed-systems-basics.md)

## 参考来源

- [DSLC 论文来源归档](../../sources/papers/dslc_multirobot_secure_control_arxiv_2609_06896.md)
- [arXiv:2609.06896 HTML 全文](https://arxiv.org/html/2609.06896v1)
- [IEEE Transactions on Robotics Early Access](https://ieeexplore.ieee.org/iel8/8860/4359257/11710271.pdf)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.06896)
