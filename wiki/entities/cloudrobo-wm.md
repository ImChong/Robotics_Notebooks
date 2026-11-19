---
type: entity
project_id: cloudrobo-wm
project: https://www.huaweicloud.com/about/live/261116.html
code: https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/CloudRobo_WM
tags: [world-models, wam, robot-learning, manipulation, robotwin, cloudrobo, huawei, flow-matching]
status: complete
updated: 2026-11-19
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./robotwin.md
  - ./dexmal-dw05.md
original:
  - ../../sources/sites/cloudrobo-wm12-2026.md
  - ../../sources/repos/cloudrobo_wm_repository.md
  - ../../sources/repos/cloudrobo_wm_xpolicylab.md
sources:
  - ../../sources/sites/cloudrobo-wm12-2026.md
  - ../../sources/repos/cloudrobo_wm_repository.md
  - ../../sources/repos/cloudrobo_wm_xpolicylab.md
  - ../../sources/models/cloudrobo-wm-1-0-robotwin-20261111.md
summary: "CloudRobo-WM 是华为云的世界动作模型家族：通用 WM 项目说明、公开 WM-1.0 RoboTwin 后训练权重与 XPolicyLab 推理/后训练适配、2026-11-16 官方发布 WM1.2。三个名称不能直接按代码或评测互相等同。"
---

# CloudRobo-WM｜华为云世界动作模型与 WM-1.0 / WM1.2 版本核验

## 一句话定义

**CloudRobo-WM** 是面向机器人操作的 World-Action Model（WAM）模型家族，联合世界未来表征与动作生成；现有可核验材料分别覆盖通用 CloudRobo-WM、WM-1.0 RoboTwin checkpoint 与官方发布的 WM1.2，必须按证据归属区分。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WAM | World-Action Model | 将世界预测与机器人动作生成结合的策略 |
| WM | World Model | 环境未来状态/视觉表征预测模型 |
| PPA | PPA（官方未在公告展开英文全称） | WM1.2 官方提出的多模态世界表征架构 |
| MoT | Mixture of Transformers | 多模态专家式动作建模 |
| RGB | Red Green Blue | XPolicyLab 适配器使用的三个彩色相机通道 |
| BF16 | Brain Floating Point 16 | 公开 checkpoint 使用的模型数值精度 |

## 为什么重要

- **世界预测与控制相连**：在传统 [VLA](../methods/vla.md) 由观测直接生成动作的基础上，强调下一步物理交互后果与动作策略的共同表征。
- **公开资产实际可操作**：XPolicyLab 有 RoboTwin 适配器、单环境在线服务、后训练入口和可校验的 WM-1.0 后训练权重，不只是论文宣传。
- **版本溯源尤其关键**：官方在 2026-11-16 介绍 WM1.2 五项技术，但当时可核实的 HF checkpoint 明确写着 WM-1.0；若混称 WM1.2，会让基准、模型部署和开源程度失真。

## 三类材料与版本关系

| 名称 | 一手材料 | 可确认内容 | 不能直接推断 |
|---|---|---|---|
| CloudRobo-WM | 个人仓库 Barry-Liang/CloudRobo-WM 的原始 README 与 PDF 入口 | 联合预测未来视觉模态与动作；README 仍写代码/数据/模型即将公开 | 仓库所属者已被华为官方认证；PDF 细节已通读 |
| CloudRobo-WM-1.0-RoboTwin-20261111 | HF 文件标识 + XPolicyLab 明确引用的模型 SHA256 | RoboTwin 后训练 checkpoint 与预训练初始化是不同文件 | 其二进制与 WM1.2 正式包相同 |
| CloudRobo-WM1.2 | 华为云 2026-11-16 官方公告、发布会议程 | 昇腾原生共享世界骨干、PPA、DualAct、异构数据监督、跨本体与弹性推理 | 独立专属权重、精确全量代码、延迟和可复现榜单均已核实 |

另外，[华为云 CloudRobo 官方文档](https://support.huaweicloud.com/cloudrobo_faq/cloudrobo_06_0005.html) 提到 **CloudRobo-WM-7B-Preview**：Wan2.2 视频骨干、Flow Matching、7B 规模、RoboTwin 2.0 Clean 演示后训练，适配 arx_x5 双臂。这是平台预览模型说明，尚无足够依据认定与上述 1.0 文件逐位一致或即为 1.2。

## 核心原理

### 通用 WM 与公开的 RoboTwin 推理路径

华为 CloudRobo 文档将模型描述为将视觉观测、机器人状态和语言指令映射为连续动作序列；原始项目 README 明确同时建模未来视觉模态与动作。对已公开 XPolicyLab 路径，可进一步核验三视角 RGB 与 14 维双臂关节状态、BF16 checkpoint、连续动作 chunk 的在线调用。

### WM1.2 官方五项技术（均为官方发布声明）

1. **PPA 世界表征**：在静态视觉识别之外强调未来物理交互预测。
2. **DualAct**：世界模型原生动作与基于 MoT 的动作专家两条实现路线，可调整专家规模。
3. **多粒度监督**：允许数据缺动作、深度、部分维度或相机视角，不必整条弃用。
4. **跨本体运动建模**：统一保留不同机器人动作语义。
5. **弹性推理**：World/Expert Action 与 Joint/Action-only 的按需组合；新闻没有提供实际每档推理时延。

### 流程总览

下图是**依据华为云公告与公开适配器接口整理的逻辑级数据流**，不是对尚未取得的 WM1.2 真实源码的逆向证明。

~~~mermaid
flowchart TD
  V[视觉观测] --> E[共享世界模型表征]
  S[机器人状态] --> E
  L[自然语言指令] --> E
  E --> P[未来物理交互表征 PPA]
  E --> D[DualAct 动作生成]
  P -. 预测信息 .-> D
  D --> A[连续动作序列]
  A --> R[机器人仿真或控制执行]
  H[异构多粒度数据] --> T[预训练监督]
  T -. 训练阶段 .-> E
  T -. 训练阶段 .-> D
~~~

图中 PPA 影响动作生成的箭头只表示官方宣称的预测增强/共享表示关系，不主张具体神经网络张量连接。WM1.2 的 World/Expert 与 Joint/Action-only 具体分支尚无可审计源码，不能伪造内部流程。

## 工程实践

| 事项 | 当前可核验路径 |
|---|---|
| 模型资产 | HF 上 WM-1.0 checkpoint，XPolicyLab 下载/验证脚本，需逐文件校验 SHA256 |
| 仿真本体 | RoboTwin 2.0 / arx_x5，三路 RGB、Joint14，动作类型 joint |
| 快速验证 | XPolicyLab 的 install.sh、smoke_test.sh --static、eval.sh |
| 推理配置 | 4 步推理、32 步重规划、seed 42 为此适配器配置，不能等同 WM1.2 官方所有模式 |
| 后训练 | XPolicyLab train.sh → tools/train_entry.py，另需检查 posttrain/README 和 Ascend 环境 |
| WM1.2 源码 | 华为云已宣布发布暨开源；版本专属可执行实现与权重映射未完成核实 |

### 源码运行时序图（仅适用于 XPolicyLab 公开的 CloudRobo_WM 适配器）

以下调用链已核对 eval.sh、setup_eval_policy_server.sh、deploy.py 与 model.py 入口；**不代表未公开的 WM1.2 原生推理代码**。

~~~mermaid
sequenceDiagram
  autonumber
  participant CLI as eval.sh
  participant S as setup_eval_policy_server.sh
  participant PS as XPolicyLab/setup_policy_server.py
  participant M as CloudRobo_WM/model.py
  participant C as setup_eval_env_client.sh
  participant Env as RoboTwin TASK_ENV
  CLI->>S: 启动策略服务器并分配端口
  S->>PS: --config_path deploy.yml
  PS->>M: 初始化 Model 和 BF16 checkpoint
  CLI->>C: 启动评测客户端
  C->>Env: 开始 episode
  loop 每轮动作
    Env-->>C: 三路 RGB + Joint14
    C->>M: update_obs(observation)
    C->>M: get_action()
    M-->>C: 动作 chunk
    C->>Env: take_action(action)
  end
  Env-->>C: 结束与成功标记
~~~

## 评测与结果（必须区分来源）

- XPolicyLab README 自报其 **CloudRobo-WM 适配器**：Clean/seen 4764/5000（95.28%），Random/unseen 4134/5000（82.68%），等权均值 **88.98%**；其 evidence / results 目录含公开记录。当前未独立运行或重算。
- 华为云 2026-11-16 新闻宣称 WM1.2 在 **RoboTwin 2.0** 和 **LiberoPro 2.0** 位列第一，但官方正文没有可逐项核验的 WM1.2 checkpoint、任务协议或完整数值。
- 两条信息不能自动合并为“已用 WM1.2 精确复现 88.98%”；同名 CloudRobo-WM 在不同榜单与提交日期上的数字也需按快照分别保存。

## 与其他工作对比

- 与 [Dexmal DW05](./dexmal-dw05.md) 同属视频/世界预测与动作联合建模方向；公开代码、许可证和版本标识应逐个对齐。
- 与 [传统 VLA](../methods/vla.md) 相比，强调未来视觉和物理交互表征参与策略学习，但不能由官方文案判断实际推理必然生成未来视频。
- 与 [世界动作模型概念页](../concepts/world-action-models.md) 对照，应区别 WAM 联合建模、独立世界模型及仅训练期辅助未来监督三种机制。

## 局限与风险

1. 尚未通读个人仓库所托管的 20261113 技术报告 PDF；其作者归属、数据集规模、真机结果和许可证不可作为已核实事实。
2. HF 模型卡全文当前不可直接访问，公开 SHA256 来自 XPolicyLab 文档；未下载 checkpoint 进行本地校验。
3. 已查看的代码是 XPolicyLab RoboTwin 适配实现，**不是 WM1.2 专属完整源码证明**。
4. 官方“第一”是厂商原始声明；缺完整模型卡、任务数、预算及独立多种子复现报告。
5. 机器人部署需检查动作单位、归一化统计、相机配置与仿真/真机反馈；不可直接把评测脚本用于高风险真机。

## 关联页面

- [World-Action Models](../concepts/world-action-models.md) — 核验 WAM 定义与边界
- [生成式世界模型](../methods/generative-world-models.md) — 世界预测表征
- [VLA 方法](../methods/vla.md) — 视觉语言到动作的对照
- [机器人操作](../tasks/manipulation.md) — 机器人任务与执行接口
- [RoboTwin](./robotwin.md) — 仿真任务与评测
- [Dexmal DW05](./dexmal-dw05.md) — Joint WAM 对照

## 参考来源

- [WM1.2 官方资料全文归档](../../sources/sites/cloudrobo-wm12-2026.md)
- [CloudRobo-WM 原始仓库归档](../../sources/repos/cloudrobo_wm_repository.md)
- [XPolicyLab 开源实现及评测归档](../../sources/repos/cloudrobo_wm_xpolicylab.md)
- [WM-1.0 模型资产索引](../../sources/models/cloudrobo-wm-1-0-robotwin-20261111.md)

## 推荐继续阅读

- [华为云 WM1.2 官方公告](https://www.huaweicloud.com/news/2026/20261116171910882.html)
- [华为云 CloudRobo 官方模型说明](https://support.huaweicloud.com/cloudrobo_faq/cloudrobo_06_0005.html)
- [XPolicyLab CloudRobo_WM README](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/CloudRobo_WM/README.md)
- [CloudRobo-WM 技术报告 PDF 入口（尚未通读）](https://github.com/Barry-Liang/CloudRobo-WM/blob/main/Cloudrobo_WM_20261113.pdf)
