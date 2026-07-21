---
type: concept
tags: [systems-engineering, edge-computing, cloud, robotics, deployment]
status: complete
updated: 2026-07-21
related:
  - ./container-orchestration-cicd.md
  - ./model-versioning-ota.md
  - ./network-protocol-stack.md
  - ./distributed-systems-basics.md
  - ./control-inference-frequency-decoupling.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md
summary: "边缘计算与云端协同：机载/边缘闭环保实时，云端负责训练、大规模仿真、队级分析与 OTA；断网为默认假设。"
---

# 边缘计算与云端协同（Edge–Cloud Robotics）

## 一句话定义

**边缘–云端协同** 把机器人能力拆到 **低延迟本地闭环** 与 **高算力云端批处理**：边缘保安全与控制，云端保学习、存储与队级优化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Edge | Edge Computing | 靠近数据源/设备的计算 |
| MEC | Multi-access Edge Computing | 多接入边缘计算 |
| WAN | Wide Area Network | 广域网 |
| CDN | Content Delivery Network | 内容分发 |
| RTT | Round-Trip Time | 往返时延 |

## 为什么重要

- VLA/大模型推理常在边或近边；1 kHz 力矩必须在机载 RT 层。
- 断网、弱网是现场常态——云不可用时机器人仍须安全站立/停机。

## 核心原理

| 层级 | 职责 | 典型技术 |
|------|------|----------|
| 机载实时 | 传感–控制–安全 FSM | RTOS、CAN/EtherCAT、LCM |
| 边缘网关 | 汇聚、缓存、协议转换、本地推理 | NUC/Jetson、容器、DDS/ROS 2 |
| 云端 | 训练、仿真 farm、分析、OTA 源 | K8s、对象存储、模型 registry |

## 工程实践

1. **数据上行**：遥测降采样 + 事件触发上传；原始 bag 可延迟同步。
2. **策略下行**：仅晋升后的签名版本；见 [OTA](./model-versioning-ota.md)。
3. **退化模式**：云/边断连 → 本地保守策略或 Passive 安全态。
4. **频率**：云推理结果经 [频率解耦](./control-inference-frequency-decoupling.md) 接到高频执行。

## 局限与风险

- 「全云控制」在 RTT 与抖动下不适合动态平衡任务。
- 边缘缓存脏版本会导致队内行为不一致——用版本号而非「latest」。

## 关联页面

- [分布式系统基础](./distributed-systems-basics.md)
- [容器编排与 CI/CD](./container-orchestration-cicd.md)
- [控制/推理频率解耦](./control-inference-frequency-decoupling.md)

## 参考来源

- [DDS/RTOS/边云/OTA/安全 FSM 一手资料](../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)

## 推荐继续阅读

- NIST 边缘/雾计算相关定义与综述
