---
type: entity
tags: [google-deepmind, vla, gemini, embodied-ai, product, hmi-papers, whole-body, on-device]
title: Gemini Robotics
status: complete
project_id: gemini-robotics
project: https://deepmind.google/models/gemini-robotics/
arxiv: "2503.20020"
summary: "Gemini Robotics 模型族持续迭代：Gemini Robotics 2.5（2026-11）在官方报告的 200 多种任务、奔驰 Apollo 2 零件分拣试点与安全引导中改进灵巧性和可靠性；VLA 仍面向合作伙伴，ER 2 与 Argon 分别承担具身编排与研发辅助。"
updated: 2026-11-11
related:
  - ./paper-roborefer.md
  - ./robospatial.md
  - ../overview/spatial-reasoning-benchmarks-technology-map.md
  - ../methods/vla.md
  - ../concepts/foundation-policy.md
  - ../concepts/whole-body-control.md
  - ../tasks/loco-manipulation.md
  - ./paper-dpc.md
  - ../overview/hub-cross-embodiment.md
  - ../methods/robotics-transformer-rt-series.md
  - ./paper-palm-e-embodied-language-model.md
  - ./gemini-4-argon.md
  - ./perceptron-egocentric.md
  - ../queries/hmi-papers-coverage.md
  - ../queries/embodied-six-routes-holes.md
original:
  - ../../sources/blogs/gemini_robotics_25_general_dexterity.md
sources:
  - ../../sources/blogs/gemini_robotics_25_general_dexterity.md
  - ../../sources/papers/gemini_robotics_arxiv_2503_20020.md
  - ../../sources/blogs/gemini_robotics_2_whole_body.md
  - ../../sources/blogs/gemini_4_argon_robotics.md
  - ../../sources/sites/gemini-robotics.md
  - ../../sources/repos/google-gemini-robotics-samples.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Gemini Robotics

**Gemini Robotics** 是 Google DeepMind 面向物理交互的 Gemini 系列机器人模型族。HMI 论文/报告总索引将其收录为 **P061**（世界模型、VLA 与 Agent）。最新发布版本为 **Gemini Robotics 2.5**（2026-11-10）；原有 **Robotics 2 / ER 2 / On-Device 2** 历史内容保留。

## 一句话定义

**Gemini Robotics**：面向物理交互的 Gemini 系列机器人模型族，由 **VLA 策略骨干**、强调空间/任务编排的 **Embodied Reasoning（ER）** agent，以及可本地运行的 **On-Device VLA** 组成；**Gemini Robotics 2** 把控制推进到人形 **feet-to-fingertips** 全身；**2.5** 将重点推进到复杂灵巧操作、生产试点可靠性和推理时物理安全引导。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言 → 电机控制的策略模型 |
| ER | Embodied Reasoning | 具身推理 / 高层 agent（VLM），编排 VLA 与工具 |
| VLM | Vision-Language Model | Gemini 多模态语言主干；ER 变体属此类 |
| DoF | Degrees of Freedom | 如 SharpaWave 手 22 DoF |
| Live API | Gemini Live API | 双向流式端点，供 ER 低延迟编排 |

## 为什么重要

- 代表闭源多模态大模型接入机器人控制的产品化叙事，常与开源 [Octo](../methods/octo-model.md) / [OpenVLA](paper-openvla.md) / [π 系列](../methods/pi07-policy.md) 对照阅读。
- **GR2 / GR2.5** 把「全身 loco-manipulation + 多指灵巧 + 多机协作 + 复杂接触任务」写进同一产品线，是产业侧对 [loco-manipulation](../tasks/loco-manipulation.md) 与学习式 [WBC](../concepts/whole-body-control.md) 的强信号。
- ER 变体把「会做动作」与「会分解长程任务」拆开；ER 2 另提供 **可调用的 API + 样例仓**，而 VLA 权重仍 gated。
- 在自动标注与评测生态中常被用作强教师/对照（见 [Perceptron Egocentric](./perceptron-egocentric.md) 对 ER-1.6 的用法）。

## 核心原理

公开材料通常强调分层：

1. **ER（高层）**：理解指令与场景 → 多步计划 → 调用 VLA / 导航 / 搜索等工具 → 用视频进度分类与 moment-finding 判断何时与完成。
2. **VLA（低层动作）**：视觉 + 语言 → 全身或双臂电机命令；同一 checkpoint 可覆盖不同手/夹爪本体（官方演示）。
3. **On-Device VLA**：本地推理 + 「motion transfer」式快速适配新双臂本体（博客：数小时、<200 例量级）。

与早期 [PaLM-E](./paper-palm-e-embodied-language-model.md) 一脉相承的是「传感器与语言共享推理上下文」，但 Gemini Robotics 更明确地把输出接到机器人执行接口。

```mermaid
flowchart TB
  H["人类指令 / 语音"] --> ER["Gemini Robotics ER 2<br/>任务跟踪与编排"]
  CAM["多相机 / 视频流"] --> ER
  ER -->|"子目标 / tool call"| VLA["Gemini Robotics 2.5 VLA<br/>（端侧另有 On-Device 2）"]
  ER -->|"导航 / Spot API 等"| API["机器人技能 API"]
  VLA --> SAFETY["GR 2.5 推理时安全引导<br/>避开物理风险"]
  SAFETY --> BODY["全身人形 / 双臂 / 末端"]
  API --> BODY
  BODY --> CAM
```

### Gemini Robotics 2 能力轴（官方博客归纳）

| 轴 | 要点 | 可信度读法 |
|----|------|------------|
| 全身控制 | Apollo 2 走取放、蹲伸、清理杂乱房间叙事 | 演示；速度自承仍不足 |
| 灵巧 | SharpaWave 打结/封袋；Robotiq 紧凑装箱 | 多指成功率方差大，博客点名仍难 |
| 多机协作 | 异构机器人分工完成单机难做的 workflow | ER 2 编排；样例偏 Spot API |
| 跨本体 | 同 checkpoint 跨 Apollo 手型与 Franka；On-Device 适配 SO101 等 | 适配数据量自报 |
| 安全 | ASIMOV-Agentic；人靠近急停；安全报告 PDF | 自报基准 + 公开 PDF |


## Gemini Robotics 2.5｜2026-11-10 官方更新

> 英文标题：**Gemini Robotics 2.5 accelerates real-world general dexterity**；以下指标均为 **Google DeepMind 官方自报**。主要资料已逐节通读，见[2026-11-10 发布归档](../../sources/blogs/gemini_robotics_25_general_dexterity.md)。

### 可核实的评测与工业试点

| 指标 / 场景 | 官方结果 | 研究与工程读法 |
|---|---|---|
| 任务覆盖 | 200 多项不同操作任务 | 食物夹取、礼盒蝴蝶结、移液等；并非统一开放 benchmark |
| 操作成功率 | 各任务类别较旧版基线提高 **30%–100%** | 相对改进范围，不是「总成功率 30%–100%」 |
| 完成速度 | 任务耗时缩短对应 **20%–30% 更快** | 不能解读为模型推理速度、控制频率提升 |
| 工业 proof-of-concept | 奔驰 Kecskemét（匈牙利）和 Sindelfingen（德国）产线，Apptronik Apollo 2 汽车配件 **kitting 成功率 96%** | ER 负责跟踪分拣流程，GR 2.5 负责双手/手臂动作；非量产级 SLA |
| 物理安全引导 | 靠近障碍物时安全任务完成表现 **+33%**；人员附近 **+43%** | 指内部评测的 safe task completion，非事故概率或安全认证 |
| Agentic 对抗安全 | **Gemini 4 Argon** 的越狱抵抗率 **99.5%**，旧 **ER 2** 基线 **82%** | **不同模型与评测维度**，不可作为 VLA 2.5 动作准确率 |

官方图表列举了夹子装糕点（5%→100%）、解开缎带蝴蝶结（0%→70%）、合上工具箱盖（0%→93%）等场景。这些是**具体单项任务**而不是通用成功率；文中没有提供完整可复现的任务数据和评测协议。

### Gemini Robotics 2.5 任务控制流程

```mermaid
flowchart TD
  CMD["任务指令 / 作业目标"] --> ER25["Gemini Robotics ER<br/>高层分拣工作流"]
  CAM25["机器人视觉观测"] --> ER25
  CAM25 --> POLICY25["Gemini Robotics 2.5 VLA<br/>视觉 + 语言到动作"]
  ER25 -->|"子任务目标"| POLICY25
  POLICY25 --> GUIDE25["推理时 reasoning guidance<br/>安全 steerability"]
  GUIDE25 --> MOVE25["Apollo 2 手臂 / 手指运动"]
  MOVE25 --> TASK25["金属支架料箱拾取与排序入盘"]
  TASK25 -->|"观测与缺漏修正"| CAM25
```

此图是按官方博客整理的**功能级信息/控制流**，不是公开实现的调用栈：官方未披露 VLA 内部网络、控制周期、低层安全控制器、ER→VLA 通信协议及统计显著性。**Gemini 4 Argon** 通过数字孪生、仿真练习和失败分析帮助**研发/训练**，不能画成在线动作控制的必经节点。

### 开源与安全评测边界（2026-11-11）

- **2.5 VLA**：仅面向机器人合作伙伴 rollout，官方产品页和博客未给出公开权重、训练/推理仓或公众 VLA API；**不能当作已开源基线**。
- **ER 2 / On-Device 2**：仍是产品线中分开的高层编排、端侧 VLA 组件；ER 2 有 API / [robotics-samples](../../sources/repos/google-gemini-robotics-samples.md)，但该样例**不是 GR 2.5 VLA 代码**。
- **ASIMOV-Jailbreak 已公开评测资源**：Google 在 [ASIMOV-Agentic Hugging Face 数据集](https://huggingface.co/datasets/google/asimov_agentic) 发布 `asimov_physical_safety_eval.py` 与 `physical_safety_duo/` 多轮对抗评测资源；这与 GR 2.5 模型权重的开放状态无关。
- **音频边界**：官方产品页明确 GR 2.5 的 VLA 输入为**视觉和语言**，任务语音交互由产品线交互 / ER 层承担；目前没有直接证据表明 GR 2.5 VLA 模型原生消费原始音频流。

## 流程总览

```mermaid
sequenceDiagram
  autonumber
  participant User as 用户
  participant ER as ER 2 (Live API)
  participant Tools as 工具层 (VLA/导航/搜索)
  participant Robot as 机器人本体
  User->>ER: 自然语言任务
  ER->>ER: 场景理解与多步规划
  loop 直到任务完成或请求人类
    ER->>Tools: tool call / 子目标
    Tools->>Robot: 电机或技能命令
    Robot-->>ER: 视频/状态反馈
    ER->>ER: 进度分类 / moment-finding / 安全检查
  end
  ER-->>User: 完成、澄清或拒绝不安全请求
```

## 源码运行时序图

**2.5 VLA / On-Device 训练与权重：不适用**（合作伙伴受限访问，未提供可运行训练 / 推理源码）；以下是**已有 ER 2 编排样例**的时序图，不能冒充 2.5 VLA 的运行代码。

**ER 2 编排样例（可运行，对齐 [`robotics-samples`](../../sources/repos/google-gemini-robotics-samples.md)）：**

```mermaid
sequenceDiagram
  autonumber
  participant UI as Web UI / 麦克风摄像头
  participant Agent as live-api/agent/server.py
  participant Gemini as Gemini Live API (ER 2)
  participant Emb as embodiment/spot|tinybot
  participant SpotAPI as live-api/spot/apps/api
  participant Spot as Boston Dynamics Spot
  UI->>Agent: 音视频 / 文本
  Agent->>Gemini: WebSocket 双向流
  Gemini-->>Agent: tool call / 文本
  Agent->>Emb: tool_call_handler 分发
  Emb->>SpotAPI: REST (导航/臂/lease)
  SpotAPI->>Spot: bosdyn-client
  Spot-->>SpotAPI: 状态 / 相机
  SpotAPI-->>Agent: 观测回灌
  Agent-->>UI: 流式回复与状态
```

关键复现路径：`live-api/agent` 用 `uv` 启动 Physical Agent Server，配置 `GEMINI_API_KEY`；Spot 侧另起 `live-api/spot` REST，再把 `--robot_url` 指过去。入门 notebook：`Getting Started/gemini_robotics_er.ipynb`。

## 工程实践

1. **能力边界与数值以官方博客 + 安全 PDF + Model Card 为准**，不要只引用二手摘要；读取日期写进笔记。
2. **开源状态（基于 2026-07-31 历史项目页 + 2026-11-11 新版本复核）**

| 组件 | 状态 | 入口 |
|------|------|------|
| ER 2 | API public preview | [AI Studio / `gemini-robotics-er-2-preview`](https://ai.dev/prompts/new_chat?model=gemini-robotics-er-2-preview)、[Robotics docs](https://ai.google.dev/gemini-api/docs/robotics-overview) |
| 编排样例 | **已开源** Apache-2.0 | [robotics-samples](https://github.com/google-gemini/robotics-samples) |
| VLA 2.5 / On-Device 2 | **未开源权重** | GR 2.5：robotics partners；On-Device 2：此前 Trusted Tester / partners |
| 安全报告 | 公开 PDF | [Gemini-Robotics-2-Safety.pdf](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-2-Safety.pdf) |

3. 与开源通用策略对比时，对齐任务定义、数据可见性与是否允许专用微调。
4. **自动标注对照：** Macrodata **WGO-Bench** 以 **Gemini 3.5 Flash + Gemini Robotics ER-1.6** 构建机器人子任务分段管线——见 [Perceptron Egocentric](./perceptron-egocentric.md)；ER 2 的 progress/moment 能力可作为后续对照候选，但勿直接替换旧数字。
5. 人形部署仍应假设需要低层跟踪 / 安全层；博客「全身 VLA」不等于取消 [WBC](../concepts/whole-body-control.md) 与接触约束栈。

## 结论

**Gemini Robotics 2.5 适合作为「闭源灵巧 VLA + ER 任务编排 + 工业试点」对照节点：其相对改进需按官方任务基线理解，ER 侧可用已有 API/样例，2.5 VLA 暂不能独立复现。**

- 读 GR2/2.5 时分开三层：ER 编排、合作伙伴 VLA、On-Device 适配——访问权限不同。
- 奔驰 96% 是零件 kitting 试点成功率；人员附近 +43% 是安全完成表现的相对改进，不是事故降低率。
- 全身与多指是宣传主轴，但官方自承 **速度与多指灵巧仍弱**；引用成功率必须回图并标「自报」。
- 跨本体叙事（同 checkpoint / 数小时适配）与 [hub-cross-embodiment](../overview/hub-cross-embodiment.md) 对照读，勿与 OXE 开源迁移混为一谈。
- 安全侧优先读 Safety PDF 与 ASIMOV-Agentic，而不是演示视频。
- 与 PaLM-E / RT / 开源 VLA 对照时分开「推理 API」与「可部署动作接口」。

## 局限与风险

- VLA / On-Device **权重与完整训练数据不可复现**；仅 ER API + 样例可工程试用。
- 产品迭代快，博客数字需标注读取日期；勿把营销演示写成学术 SOTA。
- 「同一 checkpoint 跨本体」的泛化边界（传感器、DoF、接触模态）未在公开材料中完整披露。
- HF 上 ASIMOV-Agentic 可能有访问限制；写入评测前先核对该数据集实际可见性。

## 版本沿革：1.0 → 1.5 → 2 → 2.5

| 发布 | 版本 | 可确认的变化与开放边界 |
| --- | --- | --- |
| 2025-03 | Gemini Robotics / ER | 视觉语言动作与具身推理两类模型，历史起点 |
| 2025-09-25 | Robotics 1.5 / ER 1.5 | ER 调工具并规划，VLA 执行且加入行动前推理、跨本体迁移；ER 提供 API，VLA 面向选定合作伙伴 |
| 2026-07-30 | Robotics 2 / ER 2 / On-Device 2 | 全身控制、具身 agent 与端侧模型 |
| 2026-11-10 | **Robotics 2.5** | 200 多种操作任务、奔驰 Apollo 2 零件分拣试点（96%）与推理时安全引导；ER 2 / On-Device 2 仍作为不同组件列示 |

2026-10-13 的 Google Developers 博客另介绍了通用物理推理模型 **Gemini 4 Argon** 在机器人开发中的用法。本文将其作为相邻模型单独整理在 [Gemini 4 Argon](./gemini-4-argon.md)，不把它并入 Gemini Robotics 2 的版本能力。来源：[Gemini 4 Argon for Robotics](../../sources/blogs/gemini_4_argon_robotics.md)。

**API 可用与开源权重是不同状态**。1.5 的技术说明来自[官方版本发布归档](../../sources/sites/gemini-robotics-15.md)，不把 2 的接口和实验回写为 2025 首版能力。

## 关联页面

- [VLA](../methods/vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [DPC](./paper-dpc.md) — 把 GR2 这类分层全身 VLA 写成 System 1→冻结 System 0 的三条瓶颈例（Symbiosis 2026-08）
- [跨具身迁移（知识链）](../overview/hub-cross-embodiment.md)
- [PaLM-E](./paper-palm-e-embodied-language-model.md)
- [Perceptron Egocentric](./perceptron-egocentric.md)
- [HMI 论文导读](../queries/hmi-papers-coverage.md)
- [六条路线的窟窿](../queries/embodied-six-routes-holes.md) — ER 作为高层编排、VLA 作为手的产业坐标
- [CLIFT](./paper-clift-closed-loop-iterative-finetuning.md) — 通过托管 SFT API 把 Gemini Robotics On-Device 适配成人形专才（arXiv:2607.29172）

## 参考来源

- [gemini_robotics_25_general_dexterity.md](../../sources/blogs/gemini_robotics_25_general_dexterity.md) — GR2.5 官方发布、工业试点、指标、开放边界（2026-11-10）
- [gemini_robotics_2_whole_body.md](../../sources/blogs/gemini_robotics_2_whole_body.md) — GR2 历史发布博客归档
- [gemini-robotics.md（产品页）](../../sources/sites/gemini-robotics.md)
- [google-gemini-robotics-samples.md](../../sources/repos/google-gemini-robotics-samples.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)

## 推荐继续阅读

- [Gemini Robotics 2.5 官方发布](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/) — 最新 VLA 进展、奔驰试点与安全引导
- [Gemini Robotics 2.5 产品子页](https://deepmind.google/models/gemini-robotics/vla/) — 当前产品定位和访问状态
- [Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/) — 通用物理推理模型在机器人研发工作流中的示例
- [Gemini Robotics 2 发布博客](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- [Introducing Gemini Robotics ER 2（开发者博文）](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- [Gemini Robotics 产品页](https://deepmind.google/models/gemini-robotics/)
- [ER 2 Model Card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- [Safety Technical Report (PDF)](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-2-Safety.pdf)
- [robotics-samples（GitHub）](https://github.com/google-gemini/robotics-samples)
- [Gemini Robotics 1.5 博客（前代）](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)
