---
type: overview
tags:
  - control
  - machine-learning
  - neural-network
  - gaussian-process
  - fuzzy-logic
status: complete
updated: 2026-07-18
summary: "机器学习控制离线/在线拟合残差动力学，为 PID/CTC 提供数据驱动补偿。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/neural-network-compensation-control.md
  - ../methods/gaussian-process-control.md
  - ../methods/fuzzy-logic-control.md
  - ../methods/unsupervised-clustering-fault-compensation.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 机器学习驱动控制（体系⑦）

用标注数据拟合解析模型难以描述的摩擦残差、柔性形变等，常作为传统控制器的补偿模块。

## 一句话定义

用标注数据拟合解析模型难以描述的摩擦残差、柔性形变等，常作为传统控制器的补偿模块。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GP | Gaussian Process | 非参数概率回归 |
| NN | Neural Network | 万能逼近补偿网络 |
| LSTM | Long Short-Term Memory | 时序依赖建模 |

## 为什么重要

复杂摩擦、柔性关节、非标机构难以写清解析式；NN/GP 可 **贴数据补洞**。

## 核心原理

监督学习拟合 $(q,\dot{q},\ddot{q})\mapsto \tau_{res}$；GP 给出均值与不确定度；模糊逻辑编码专家规则；无监督聚类识别工况切换补偿表。

## 代表性算法

| 算法 | 节点 |
|------|------|
| NN 补偿 | [neural-network-compensation-control.md](../methods/neural-network-compensation-control.md) |
| GP | [gaussian-process-control.md](../methods/gaussian-process-control.md) |
| 模糊逻辑 | [fuzzy-logic-control.md](../methods/fuzzy-logic-control.md) |
| 聚类补偿 | [unsupervised-clustering-fault-compensation.md](../methods/unsupervised-clustering-fault-compensation.md) |

## 工程实践

采集覆盖工况的轨迹数据；残差网络与解析控制器 **并联**；GP 用于小样本安全约束控制。

## 局限与风险

依赖数据分布；外推差；与 RL 不同 **不需环境试错** 但需标注或仿真标签。

## 关联页面

- [Neural Network Compensation](../methods/neural-network-compensation-control.md)
- [Neural Feedback Controller](../concepts/neural-feedback-controller.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

