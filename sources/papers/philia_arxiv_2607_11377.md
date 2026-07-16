# Philia：A Glimpse into Long-term Physical Coexistence with Intelligent Robots（技术报告）

> 来源归档（ingest）

- **标题：** A Glimpse into Long-term Physical Coexistence with Intelligent Robots（PHILIA）
- **类型：** paper（技术报告）
- **arXiv abs：** <https://arxiv.org/abs/2607.11377>
- **项目页：** <https://www.astribot.com/research/Philia>
- **机构：** Astribot Team（星尘智能）
- **平台：** 当前实现面向 **Astribot S1**；网关契约可扩展异构机器人
- **入库日期：** 2026-07-16
- **一句话说明：** **Agent–Runtime 解耦** 的多机器人助手：**OpenClaw** 控制平面 + **Robot Gateway** 能力清单；五段式 **capability-grounded dispatch**；记忆只经规划间接影响物理执行；策略以 capability 暴露并支持 **SFT → 部署 rollout + 人工干预 → advantage-conditioned 离线后训练** 与 **训练时 RTC + 运行时轨迹滤波**。

## 核心摘录（面向 wiki 编译）

### 1) 系统抽象与架构

- **链接：** §2 PHILIA
- **摘录要点：** 三层：**UI → Agent Control Plane → Robot Gateways**。助手负责意图、记忆、工具、规划；网关发布 **runtime manifest**（能力 ID + schema）；物理动作须经 **授权 / 确认 / 就绪 / 仲裁** 闸门；状态分层——控制平面持 **语义记忆**，机器人侧持 **高频执行态**（地图、传感器、策略 buffer）。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — 架构与安全信封
  - [Hermes Agent](../../wiki/entities/hermes-agent.md) — 网关与会话模型对照

### 2) Capability-grounded Dispatch

- **链接：** §3.1–3.2
- **摘录要点：** **N=483** 话语四分类（robot_task / robot_query / robot_control / non_robot）；轻量二分类门 **F1≥0.96**；前沿模型 capability router **top-1 skill 89–94%**、non-robot **拒绝率最高 100%**；五段级联：**regex → 本地分类器(414ms) → 确定性路由 → grounded router → agent turn**，部署 **Qwen3.5-4B 预滤 + claude-opus-4.6 全路由**。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — Dispatch 表与延迟分解

### 3) Policies as Capabilities

- **链接：** §2.3
- **摘录要点：**
  - **零样本基础模型**（含 Lumo 系）经统一接口即插即用；
  - **任务 SFT** 为灵巧/长程任务收集专家示范；
  - **部署闭环：** rollout + 人工纠正 → 帧级 progress advantage → **chunk-aware 二值标签** → 条件策略离线提取（非在线 PG）；
  - **平滑执行：** 训练 **RTC**（随机推理延迟 + 已执行前缀条件化）+ 部署 **轨迹滤波**（参考 Human-Level Intelligence 路线）。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — 策略演进管线
  - [Lumo-2](../../wiki/entities/lumo-2.md) — 上游通才策略
  - [Action Chunking](../../wiki/methods/action-chunking.md) — chunk 边界对齐

### 4) 记忆与导航

- **链接：** §2.4–2.5
- **摘录要点：** 长期记忆继承 **OpenClaw Markdown**（MEMORY / 日志按日）；检索 on-demand，**不直接下发电机命令**。导航作为 gateway capability：站点建图 + 语义地点标注 + 每会话 anchor 重定位；多机共享 **地图帧语义地点**、各自独立定位运行时。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — 记忆→规划→网关路径
  - [Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md) — 语义地点 vs 低层 SLAM 分界

### 5) Playbook 与真机场景

- **链接：** §3.3
- **摘录要点：** **推理接地**（观测/偏好/热量/食物类别）、**记忆接地**（早餐三明治+浓缩咖啡偏好）、**多机协同**（Alice 清桌 + Bob 提垃圾袋）、**策略即插即用升级**；摘要提及 **pack the backpack**、**lift the garbage bag** 等开放家务。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — Playbook 与项目页视频对照
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 多机服务任务语境

## 当前提炼状态

- [x] arXiv HTML 摘要 + 架构/实验主文已对齐摘录
- [x] 与 [Astribot Philia 项目页](../sites/astribot-philia-project-page.md) 视频索引交叉引用
