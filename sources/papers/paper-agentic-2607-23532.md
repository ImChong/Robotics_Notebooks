# Mission-Level Runtime Assurance for LLM-Assisted ISR Swarms over a Verification-Aware Fabric

- **作者：** Nikolaos Kekatos, Michael Ioannou, Panagiotis Katsaros, Alexios Lekidis, Theodoros Nestoridis, Tom Nianios, Dimitrios Nikou
- **机构：** Clone Systems (Larnaca, Cyprus); Bolton Technologies (Limassol, Cyprus); Aristotle University of Thessaloniki (Greece); University of Thessaly (Larissa, Greece); International Hellenic University (Greece)
- **版本 / 载体：** arXiv:2607.23532v3 [cs.CR]（v1 提交于 2026-07-26，v3 修订于 2026-10-11）；arXiv HTML（LaTeXML）
- **原文：** <https://arxiv.org/abs/2607.23532> / <https://arxiv.org/html/2607.23532>
- **项目页 / 代码：** 宣称将开源或未列链接（arXiv:2607.23532v3 附录 0.A 描述了 reproducible core `swarm_rv`、`console/`、`ros2_realism/`、`sitl_mission/` 等复现脚本及 Docker 打包方式，但截至 2026-11-27 论文正文及元信息检索均未附公开 GitHub / 项目页链接，亦无 `candidate_repos` 可供静态核查）
- **核对范围：** 通读 arXiv:2607.23532v3 HTML 全文（Abstract、§1–§9、Table 1–9、Figure 1–7 与 Appendix 0.A Reproducibility，共 1413 行文本）；静态核对论文文本，未本地运行仿真或外部代码
- **通读：** 已完成

## 原文核对笔记

### 1. 问题定义与威胁模型（§1–§3, Table 1）

- 在对抗通信环境下，LLM 辅助的情报监视与侦察（ISR）无人系统集群面临两类叠加风险：
  1. **跨平台组合式违规**：攻击者通过检索数据中的间接提示注入（indirect prompt injection）诱导 LLM 规划器将禁止的任务拆分为多个单机合规子动作（如 `uav_1` 采坐标、`uav_2` 拍图像、`uav_3` 采通信元数据、`ugv_1` 采进出路线），每台平台均未越过本地传感器或发射上限，但联合结果构成被禁止的完整目标情报包（$P_{\text{pkg}}$）、超出机群总发射控制（EMCON）预算（$P_{\text{emcon}}$）或违反跨机授权时序（$P_{\text{auth}}$）。
  2. **通信对抗掩盖证据**：丢包、重排或压制干扰使中心聚合器收不到关键见证事件；若监控器把“未收到违规证据”直接当作“无违规放行（all-clear）”，就会发出静默漏报。
- **信任边界（Table 1）**：核心模型假设提示注入仅劫持 LLM 规划器，受信任的事件生成与认证链路不被伪造，监控节点和逻辑不被篡改；主要防御跨平台任务拆分与通信丢包/延迟/乱序/静默干扰（单平台虚假自报归入 §7.4 探索性扩展，依赖独立 RF-sentry 旁证）。

对 wiki 的映射：[Mission-Level Runtime Assurance for LLM-Assisted ISR Swarms over a Verification-Aware Fabric](../../wiki/entities/paper-agentic-2607-23532.md)。

### 2. 规范事件流、跨智能体规约与二轴裁决代数（§3–§4, Table 2）

- **规范事件元组**：$e=(\mathit{id}, \tau, a, k, \mathit{attrs})$，包含持久化事件编号 $\mathit{id}$（用于去重与缺口检测）、任务时钟时间戳 $\tau$、平台标识 $a\in A$、事件类型 $k\in\{\text{collect}, \text{transmit}, \text{enter}, \text{authorise}, \text{observe}, \text{tick}\}$ 与属性。
- **分层属性拆解（Table 2）**：
  - **L1（单平台）**：空间驻留 $\varphi_S$（禁区进入）、一阶过去时序义务 $\varphi_T$（本地预算与任务合规）、文本语义谓词 $\varphi_\Sigma$（合法传感器与接收方）。
  - **L2（小队协调器）**：小队内部分情报聚合（向 L3 传递）、联合区域占用数 $\le k$、重复任务分配检查。
  - **L3（任务级跨智能体）**：禁止联合情报包 $P_{\text{pkg}}(F,W)$、机群 EMCON 发射总预算 $P_{\text{emcon}}(W)$、先授权后进入因果序 $P_{\text{auth}}(a_j,Z,t)$、区域覆盖 `once` 与子群静默告警。
- **二轴证据感知裁决 $(s, c)$**：安全轴 $s\in\{\textsf{violation}, \textsf{no\_violation}, \textsf{unknown}\}$，完整性轴 $c\in\{\textsf{sound}\succ\textsf{degraded}\succ\textsf{incomplete}\succ\textsf{unavailable}\}$。跨智能体性质的完整性取贡献平台证据的格下确界（meet）；一旦非 `sound`（如出现 $\mathit{id}$ 缺口、消费序违规、高水位截断或任务时钟判静默），原本的 `no_violation` 必须降级为 `unknown`，避免静默误放行。针对单调见证性质（monotone witness properties），给出条件可靠性质（conditional soundness）。

对 wiki 的映射：[Mission-Level Runtime Assurance for LLM-Assisted ISR Swarms over a Verification-Aware Fabric](../../wiki/entities/paper-agentic-2607-23532.md)。

### 3. 三层分布式架构与 RV-Fabric 五项保障（§5, Table 3, Fig. 3）

- 将作者此前边云/物联网（edge-IoT）三层监控架构（设备→网关→后端）迁移为机群语义：**平台 L1 → 小队协调器 L2 → 任务监控器 L3**。
- **五项传输与时序保障（Table 3）**：
  1. **G1 至少一次投递**：MQTT QoS 1 + L2 持久化发件箱（durable outbox），结合 `event-id` 缺口检测丢包；
  2. **G2 单调序**：协调器边车（sidecar）赋予可信单调任务摄入时间戳；
  3. **G3 静默活性**：后端常驻 1 Hz 任务时钟（mission clock），即便平台全部被干扰静默也能观测缺失流；
  4. **G4 消费者隔离**：NATS JetStream 持久化每消费者游标（durable per-consumer cursors），慢消费者不阻塞任务裁决；
  5. **G5 背压流控**：JetStream 流控与 `discard=new` 溢出告警策略。

### 4. 仿真环境、真实性层与核心评测结果（§6–§7, Table 4–9, Fig. 4–7）

- **双层实现（§6）**：确定性 Python 运动学核心（含故障注入与 stdlib 交互台 `console/`）+ 真实性层（ROS 2 Humble DDS 无头容器 4 节点发布 `/<ns>/action` + ArduPilot SITL 四旋翼 MAVLink/TCP 生成 360 个位置采样 + 真实 Mosquitto/NATS JetStream 系统实测 + 真实 MonPoly MFOTL 引擎验证）。
- **关键量化指标**：
  - 主故障战役（Table 4，$|I^{\star}|=3$）：每平台护栏检出 `0/3`（9 次静默误放行）；中心尽力而为监控器在无故障/丢见证/干扰平台下检出 `3/3`、`2/3`、`1/3`（3 次静默误放行）；RV-Fabric 分别为 `3/3`、`2/3 + 1u`、`1/3 + 2u`（`0` 次静默误放行）。
  - 跨 4 类目标汇总（Table 5）：三类故障战役下静默误放行总计——每平台 `18`、中心尽力而为 `9`、RV-Fabric `0`。
  - 溯源与扩展（§7.3–§7.5）：地面真值平台溯源匹配 `6/6 = 100%`；500 次随机任务切分实验（seed 7）下 RV-Fabric 静默放行率 `0%`（中心基线 `100%`）；4/16/40/100 平台亚毫秒级评估，1000 平台评估耗时 `1.2 ms`（Fig. 7 显示千平台以内低于 `2 ms`）；实测真实 Mosquitto + JetStream（`n=1000` 事件）无故障端到端中位延迟 `3.7 ms`，10% 丢包重传中位 `42 ms`（`1000/1000` 送达）；Table 9 中 RV-Fabric 的无静默放行率（NSCR）为 `100%`（8 小队同时故障时静默放行 `0` 次，中心基线 NSCR 为 `50%` 且静默漏报 `8` 次）。
- **局限（§7.6）**：定量核心基于确定性运动学任务（验证不变量保持而非自然分布检出率）；ROS 2/DDS 前端跳点尚未与真实 Mosquitto+JetStream 端到端链路合一；Gazebo GPU 视觉渲染与大规模真机实验留待未来工作；未列公开代码链接。
