# ROS 2 Humble 官方 Concepts 章节通读（节点 / 话题 / 服务 / 动作 / 参数 / 发现 / QoS）

> 来源归档（原文通读 · 支撑 `wiki/concepts/ros2-basics.md`）

- **标题：** ROS 2 Documentation (Humble) · Concepts：Basic（Nodes、Topics、Services、Actions、Parameters、Discovery）与 Intermediate（About Quality of Service Settings）
- **类型：** site / official-docs
- **原文：** <https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html>（同目录 About-Topics / About-Services / About-Actions / About-Parameters / About-Discovery）；<https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html>（2026-11 抓取 HTML 全文）
- **一句话说明：** ROS 2 通信原语（话题、服务、动作、参数）、自动发现与 QoS 兼容性规则的官方定义。
- **核对范围：** 逐页通读上述 7 页全文（含 QoS 页的策略列表、预设 profile、各策略兼容性表、QoS events）。未阅读 Domain ID 页后半（平台端口细节）、Launch / tf2 / Lifecycle / ros2_control 等其他章节，也未读 Science Robotics 2022 ROS 2 论文；wiki 页中这些部分仍依赖页面既有内容与其他归档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **Node：** ROS graph 的参与者，通过客户端库通信；通常是计算单元，"每个节点只做一件逻辑上的事"；可在同进程、跨进程、跨机器通信；可同时是 publisher / subscriber / service server&client / action server&client，并带可配置参数；节点间连接经分布式发现建立。C++ 另有 sub-node（共享父节点 context 与图身份，避免额外发现开销）。
- **Topic：** 用于连续数据流（传感器、状态）；匿名、强类型的发布/订阅"总线"；同名 topic 上可有零或多个 publisher / subscriber；订阅者一般不关心数据来自哪个发布者；`ros2 bag record` 只是再加一个订阅者，不打断其他数据流。强类型含两层：字段类型被强制，核心类型的字段语义明确（如 IMU 角速度单位 rad/s）。
- **Service：** 即远程过程调用；`request --- response`；应快速返回，不要用于需被抢占的长任务（改用 action）；每个 service 名只应有一个 server（多个则未定义由谁响应），client 数量不限。
- **Action：** 带反馈、可取消/抢占的长时 RPC；`request --- result --- feedback`；有建立与监控连接的开销，故用于长任务；同样每个 action 名只应有一个 server。
- **Parameter：** 属于单个节点，生命周期与节点绑定；key + value + descriptor；类型限 bool/int64/float64/string 及其数组、byte[]；默认须先声明，运行时改类型默认失败（防误配）；两类回调：set-parameter 回调（可拒绝，须无副作用）与 on-parameter-event 回调（变更成功后触发）；默认自动生成 describe/get/list/set/set_atomically 等参数服务，`ros2 param` 即基于它们。
- **Discovery：** 节点启动时在同一 `ROS_DOMAIN_ID` 内广播自己，其他节点应答；周期性重播；下线时通告；**仅在 QoS 兼容时才建立连接**。
- **QoS 策略：** History（keep last N / keep all）、Depth、Reliability（best effort / reliable）、Durability（transient local / volatile）、Deadline、Lifespan、Liveliness（automatic / manual by topic）、Lease Duration。可做到"像 TCP 一样可靠或像 UDP 一样尽力"；即使 reliable 在 ROS 2 中也基于 UDP 实现。
- **预设 profile：** 默认 publisher/subscription = keep last、深度 10、reliable、volatile；services 为 reliable + volatile（避免重启后收到过期请求的副作用）；sensor data 为 best effort + 小队列；parameters 用大队列深度；system default 取 RMW 实现默认值。
- **QoS 兼容性（Request vs Offered）：** 订阅方请求"最低可接受质量"，发布方提供"最高可提供质量"，任一策略订阅方要求高于发布方提供则不连接。例：发布 best effort + 订阅 reliable 不兼容；发布 volatile + 订阅 transient local 不兼容；deadline / lease：发布方 x、订阅方 y 须 y ≥ x；liveliness 发布 automatic + 订阅 manual 不兼容。不兼容时不传消息，可用 QoS events（offered/requested incompatible QoS、deadline missed、liveliness lost/changed）检测。ROS 1 中同名同类型一律连接，这是新坑。

## 对 wiki 的映射

- [`wiki/concepts/ros2-basics.md`](../wiki/concepts/ros2-basics.md)

## 参考来源

- <https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html>
- <https://docs.ros.org/en/humble/Concepts/Basic/About-Topics.html>
- <https://docs.ros.org/en/humble/Concepts/Basic/About-Services.html>
- <https://docs.ros.org/en/humble/Concepts/Basic/About-Actions.html>
- <https://docs.ros.org/en/humble/Concepts/Basic/About-Parameters.html>
- <https://docs.ros.org/en/humble/Concepts/Basic/About-Discovery.html>
- <https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html>
