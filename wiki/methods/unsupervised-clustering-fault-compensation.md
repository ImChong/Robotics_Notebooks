---
type: method
tags:
  - control
  - unsupervised-learning
  - clustering
  - fault-compensation
status: complete
updated: 2026-07-18
summary: "无标签工况聚类 + 补偿表切换，应对摩擦漂移与磨损。"
related:
  - ../overview/robot-control-paradigm-ml-driven-control.md
  - ./neural-network-compensation-control.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Unsupervised Clustering Fault Compensation（无监督聚类故障补偿）

聚类故障补偿：对运行数据无监督聚类识别工况/磨损模式，切换预存补偿参数，实现被动自适应。

## 一句话定义

> 聚类故障补偿：对运行数据无监督聚类识别工况/磨损模式，切换预存补偿参数，实现被动自适应。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| K-means | K-means Clustering | 常用聚类 |
| Anomaly | Anomaly Regime | 异常工况簇 |
| Comp | Compensation Map | 簇对应补偿表 |

## 为什么重要

长期运行机器人 **摩擦漂移** 难以手工重标定时，聚类是轻量维护手段。

## 核心原理

特征 $(\tau,\dot{q},e)$ 聚类 → 簇 $c$ 选补偿 $\Delta\tau_c$ 或增益表；在线最近簇分类。

## 工程实践

离线建簇与补偿表；在线监测簇迁移触发维护；与 ILC 批次数据结合。

## 主要技术路线

### 1. 工况聚类 + 补偿表

文内代表实现路径；详见 [关联概念/形式化](../concepts/system-identification.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

簇边界模糊时误切换；需足够运行数据覆盖工况。

## 关联页面

- [system-identification](../concepts/system-identification.md)
- [Neural Network Compensation](./neural-network-compensation-control.md)
- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

