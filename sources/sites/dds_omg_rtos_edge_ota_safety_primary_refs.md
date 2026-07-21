# DDS、RTOS、边缘云协同、OTA 与安全状态机一手资料索引

> 来源归档（ingest）

- **标题：** OMG DDS、实时调度/RTOS、边缘–云协同、模型版本/OTA、故障安全 FSM 一手依据
- **类型：** standard / official docs / paper（合集）
- **入库日期：** 2026-07-21
- **一句话说明：** 补齐机器人系统工程中「中间件底层、实时 OS、边云部署、模型发布、故障安全」五块与运控强相关的独立节点依据。
- **沉淀到 wiki：** 是 → [dds-communication](../../wiki/concepts/dds-communication.md)、[rtos-realtime-scheduling](../../wiki/concepts/rtos-realtime-scheduling.md)、[edge-cloud-robotics](../../wiki/concepts/edge-cloud-robotics.md)、[control-inference-frequency-decoupling](../../wiki/concepts/control-inference-frequency-decoupling.md)、[model-versioning-ota](../../wiki/concepts/model-versioning-ota.md)、[robot-safety-state-machine](../../wiki/concepts/robot-safety-state-machine.md)

## 核心摘录

### 1) OMG DDS 与 ROS 2 RMW

- **来源：**
  - [OMG DDS 1.4](https://www.omg.org/spec/DDS/1.4)（DCPS、QoS）
  - [OMG DDSI-RTPS](https://www.omg.org/spec/DDSI-RTPS/)（线协议，常跑 UDP）
  - ROS 2：[About different DDS/RTPS vendors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Different-Middleware-Vendors.html)
  - 实现：[Fast DDS](https://fast-dds.docs.eprosima.com/)、[Cyclone DDS](https://cyclonedds.io/)
- **要点：**
  - **数据中心化发布订阅（DCPS）**：Topic + Type + QoS；发现去中心化。
  - **QoS**：Reliability、History、Durability、Deadline、Liveliness——决定「要最新」还是「要必达」。
  - ROS 2 通过 **RMW** 换 vendor；默认实现随发行版变化，调优须落到具体 DDS。
- **对 wiki 的映射：** [dds-communication](../../wiki/concepts/dds-communication.md)、[ros2-basics](../../wiki/concepts/ros2-basics.md)

### 2) RTOS 与实时调度

- **来源：**
  - [FreeRTOS 内核文档](https://www.freertos.org/Documentation/RTOS_book.html)（任务、优先级、队列）
  - POSIX.1b 实时扩展（`SCHED_FIFO`、时钟、信号量）
  - Linux [PREEMPT_RT](https://wiki.linuxfoundation.org/realtime/start)
  - Liu & Layland（1973）速率单调调度思想
- **要点：**
  - **硬实时**：错过截止即失败（力矩环）；**软实时**：偶发逾期可接受（视觉）。
  - 电机 MCU 常跑 FreeRTOS/裸机；主控常跑 PREEMPT_RT Linux + 隔离核。
- **对 wiki 的映射：** [rtos-realtime-scheduling](../../wiki/concepts/rtos-realtime-scheduling.md)、[real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md)

### 3) 边缘计算与云端协同

- **来源：** NIST [*The NIST Definition of Fog Computing*](https://csrc.nist.gov/publications/detail/sp/500-325/final) / 边缘计算概述；ROS 2 / 工业实践中的 edge gateway 模式
- **要点：** 边缘做低延迟感知–控制闭环；云端做训练、大规模仿真、队级分析与 OTA；带宽/断网是一等公民约束。
- **对 wiki 的映射：** [edge-cloud-robotics](../../wiki/concepts/edge-cloud-robotics.md)

### 4) 控制频率与推理频率解耦

- **来源：** 本库 [控制环路延迟建模](../../wiki/formalizations/control-loop-latency-modeling.md)；常见分层架构（高频 PD/WBC + 低频策略/VLA）；Yobotics E3 模板等双线程实践（约 50 Hz 推理 / 500 Hz 发令）
- **要点：** 推理慢不等于控制慢——用零阶保持、动作块、轨迹缓冲或 residual PD 把低频动作接到高频执行。
- **对 wiki 的映射：** [control-inference-frequency-decoupling](../../wiki/concepts/control-inference-frequency-decoupling.md)

### 5) 模型版本管理与 OTA

- **来源：**
  - [RAUC](https://rauc.io/) / [Mender](https://mender.io/) / A/B 分区更新惯例
  - MLOps：模型 registry（版本、谱系、签名）
  - Android A/B OTA 思想（回滚槽）
- **要点：** 固件 OTA 与 **策略权重 OTA** 应分通道；必须可回滚、签名验签、与安全 FSM 联动（更新中禁止高力矩）。
- **对 wiki 的映射：** [model-versioning-ota](../../wiki/concepts/model-versioning-ota.md)

### 6) 硬件/通信故障与安全状态机

- **来源：**
  - 功能安全思想：ISO 13849 / IEC 61508（故障检测 → 安全状态）
  - 机器人部署实践：[wbc_fsm](../../sources/repos/wbc_fsm.md) Passive/Loco/WBC 模式切换
  - 本库 [Safety Filter](../../wiki/concepts/safety-filter.md)
- **要点：** 看门狗、总线超时、驱动器错误码 → 确定性 FSM 转移到阻尼/无力矩/站立保护；网络选主不能替代本地安全逻辑。
- **对 wiki 的映射：** [robot-safety-state-machine](../../wiki/concepts/robot-safety-state-machine.md)、[wbc-fsm](../../wiki/entities/wbc-fsm.md)

## 当前提炼状态

- [x] 摘要与 wiki 映射
