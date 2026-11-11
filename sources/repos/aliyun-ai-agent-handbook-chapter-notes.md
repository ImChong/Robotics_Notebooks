# AI Agent Handbook — 30 章逐章阅读记录（固定版本）

- **一手仓库：** https://github.com/aliyun/ai-agent-handbook
- **固定版本：** `a1dd02ec080c4e18930660c0154da1ef589df639`（2026-11-11）
- **覆盖范围：** 编号第 01–30 章（第 09、17、18 章各拆分为两个正文；第 25–28 章收录多篇案例），另附 2026 开发者调研和专题资料。
- **性质：** 本仓库的结构化阅读摘记与可复核章节索引；不是对官方全文的转载，不能作为独立性能评测报告。
- **图片/视频与产品控制台：** 本次仅依据版本固定的官方 Markdown 文本进行知识归纳，不声称独立核验图内数值、线上产品行为或案例评测数据。
- **上下游：** [官方原始资料归档](aliyun-ai-agent-handbook.md) · [知识节点](../../wiki/entities/aliyun-ai-agent-handbook.md)

## 一、架构篇（01–02）

### 第 01 章 AI 原生应用的新阶段

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/01-architecture/%E7%AC%AC%2001%20%E7%AB%A0%E3%80%80AI%20%E5%8E%9F%E7%94%9F%E5%BA%94%E7%94%A8%E7%9A%84%E6%96%B0%E9%98%B6%E6%AE%B5.md)
- **技术主线：** Chat/RAG、Workflow、Copilot、Single-Agent、Long-Horizon 与 Multi-Agent 的控制权和状态边界不同；Model + Harness 共同构成 Agent。长程是时间维度，多 Agent 是组织维度，二者不是单向替代链。
- **边界与复核：** 任务路径可穷举时优先 Workflow；不可逆外部动作需确定性校验与人工审批。

### 第 02 章 Agentic Application 参考架构

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/01-architecture/%E7%AC%AC%2002%20%E7%AB%A0%E3%80%80Agentic%20Application%20%E5%8F%82%E8%80%83%E6%9E%B6%E6%9E%84.md)
- **技术主线：** 组件视图（能力是什么）、平台视图（在哪里/谁负责）、生命周期视图（如何演进）正交；广义 Harness 在工程层细分为编排、Runtime、Sandbox、Observability、Governance 等。
- **边界与复核：** 优化只提出候选变更，必须回到构建、版本化、治理准入后才可进入生产；Context 不是 State。

## 二、构建篇（03–06）

### 第 03 章 Harness 的主流构建方式和责任边界

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/02-build/%E7%AC%AC%2003%20%E7%AB%A0%E3%80%80%E8%8C%83%E5%BC%8F%EF%BC%9AHarness%20%E7%9A%84%E4%B8%BB%E6%B5%81%E6%9E%84%E5%BB%BA%E6%96%B9%E5%BC%8F%E5%92%8C%E8%B4%A3%E4%BB%BB%E8%BE%B9%E7%95%8C.md)
- **技术主线：** 四种入口：高代码框架、产品化 Harness（Coding Agent CLI/SDK）、Managed Agents、云产品构建；Agent Platform 负责多源接入与规模化交付，并非第五种起点。
- **边界与复核：** 选择依据是定制深度、数据边界与运行责任；框架 API 可用不等于业务已可验收。

### 第 04 章 任务：编排、长程推进与协作流转

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/02-build/%E7%AC%AC%2004%20%E7%AB%A0%E3%80%80%E4%BB%BB%E5%8A%A1%EF%BC%9A%E7%BC%96%E6%8E%92%E3%80%81%E9%95%BF%E7%A8%8B%E6%8E%A8%E8%BF%9B%E4%B8%8E%E5%8D%8F%E4%BD%9C%E6%B5%81%E8%BD%AC.md)
- **技术主线：** Agent Loop 按 Prepare→Model→Act→Observe→Verify 推进；外置 Plan/Todo 和显式状态机管理预算、受控委派、暂停、恢复及异步任务。
- **边界与复核：** 模型停止或子 Agent 汇报完成≠最终完成；由独立 Verifier 根据产物、测试、审批与环境终态验收。

### 第 05 章 信息：上下文、状态与可复用能力资产

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/02-build/%E7%AC%AC%2005%20%E7%AB%A0%E3%80%80%E4%BF%A1%E6%81%AF%EF%BC%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E3%80%81%E7%8A%B6%E6%80%81%E4%B8%8E%E5%8F%AF%E5%A4%8D%E7%94%A8%E8%83%BD%E5%8A%9B%E8%B5%84%E4%BA%A7.md)
- **技术主线：** Context Builder 根据可信度、相关性、Token 预算编译当轮信息，压缩前应将权威事实写入外置 Task State/Workspace；Memory、Knowledge、Skill 分别治理。
- **边界与复核：** 不能以长窗口替代持久事件和版本化资产；压缩/召回后必须重建并验证上下文。

### 第 06 章 行动：受控执行、验证反馈与交付准备

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/02-build/%E7%AC%AC%2006%20%E7%AB%A0%E3%80%80%E8%A1%8C%E5%8A%A8%EF%BC%9A%E5%8F%97%E6%8E%A7%E6%89%A7%E8%A1%8C%E3%80%81%E9%AA%8C%E8%AF%81%E5%8F%8D%E9%A6%88%E4%B8%8E%E4%BA%A4%E4%BB%98%E5%87%86%E5%A4%87.md)
- **技术主线：** 统一 Action Plane 按 Schema、身份、Policy、审批、Sandbox 执行、Observation、Trace 推进；Function Calling/MCP/A2A 解决不同层次的接口问题。
- **边界与复核：** 模型提出动作、协议校验成功或工具返回 200 均不等于获得授权或达成业务结果。

## 三、运行篇（07–12）

### 第 07 章 Agent 运行时与沙箱

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2007%20%E7%AB%A0%E3%80%80Agent%20%E8%BF%90%E8%A1%8C%E6%97%B6%E4%B8%8E%E6%B2%99%E7%AE%B1.md)
- **技术主线：** 可编程 Workspace、Shell/Browser/Computer Use 运行环境与 Task 生命周期分离；隔离、快照、恢复、租约与计算回收是稳定长任务底座。
- **边界与复核：** 执行进程重启、重复调用和外部动作结果未知需对账，不能盲目重放产生重复副作用。

### 第 08 章 Agent 状态存储与语义资产

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2008%20%E7%AB%A0%E3%80%80Agent%20%E7%8A%B6%E6%80%81%E5%AD%98%E5%82%A8%E4%B8%8E%E8%AF%AD%E4%B9%89%E8%B5%84%E4%BA%A7.md)
- **技术主线：** Event Log、Checkpoint、Workspace/Artifact、长期 Memory、RAG Knowledge 和业务本体具有不同事实权威、查询方式与一致性要求。
- **边界与复核：** 不能把“向量库里保存了内容”当作任务事实持久化；要区分新鲜度、租户隔离、更新和遗忘。

### 第 09 章 AI 网关与统一流量治理

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2009%20%E7%AB%A0%E3%80%80AI%20%E7%BD%91%E5%85%B3%E4%B8%8E%E7%BB%9F%E4%B8%80%E6%B5%81%E9%87%8F%E6%B2%BB%E7%90%86/01%20AI%20%E7%BD%91%E5%85%B3%E4%B8%8E%E7%BB%9F%E4%B8%80%E6%B5%81%E9%87%8F%E6%B2%BB%E7%90%86%E7%BB%BC%E8%BF%B0.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2009%20%E7%AB%A0%E3%80%80AI%20%E7%BD%91%E5%85%B3%E4%B8%8E%E7%BB%9F%E4%B8%80%E6%B5%81%E9%87%8F%E6%B2%BB%E7%90%86/02%20%E6%99%BA%E8%83%BD%E8%B7%AF%E7%94%B1%EF%BC%9A%E5%A6%82%E4%BD%95%E5%88%A9%E7%94%A8%E4%BE%BF%E5%AE%9C%E6%A8%A1%E5%9E%8B%E6%8F%90%E5%8D%87%E6%99%BA%E4%BB%B7%E6%AF%94.md)
- **技术主线：** 本章分两篇：LLM、MCP、Agent 网关分别管理推理流量、能力调用和任务入口；HiRoute 的 Jev 智能路由专题强调阶段能力选择与总任务成本。
- **边界与复核：** 一次 API 返回成功≠Agent Task 完成；网关账单/任务口径应绑定身份、版本与完整执行轨迹。

### 第 10 章 Agent 异步任务与自动化流程

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2010%20%E7%AB%A0%E3%80%80Agent%20%E5%BC%82%E6%AD%A5%E4%BB%BB%E5%8A%A1%E4%B8%8E%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B5%81%E7%A8%8B.md)
- **技术主线：** 消息与调用通道、持久 Task、触发器、定时器、Workflow 各有语义；异步任务要求稳定 ID、状态、重试与等待人工授权。
- **边界与复核：** 超时、重投递与客户端断连不代表操作未发生；优先幂等、租约与业务结果核对。

### 第 11 章 Multi-Agent 协作与编排

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2011%20%E7%AB%A0%E3%80%80Multi-Agent%20%E5%8D%8F%E4%BD%9C%E4%B8%8E%E7%BC%96%E6%8E%92.md)
- **技术主线：** 监督者、平行工作者、层级团队依任务可分解性选择；委派要明确能力、身份、共享状态、返回 Artifact 和验收责任。
- **边界与复核：** 多 Agent 不是模型调用数竞赛；跨代理共享权限和上下文会放大污染及失败传播。

### 第 12 章 Agent 分布式通信

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/03-run/%E7%AC%AC%2012%20%E7%AB%A0%E3%80%80Agent%20%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%9A%E4%BF%A1.md)
- **技术主线：** 能力、协作、内部、人机四类通信面，对应请求应答、流式、持久任务句柄和长连接等交互语义。
- **边界与复核：** A2A 与 MCP 不能互相代替；消息是否送达、执行是否成功、业务是否结束必须分别判定。

## 四、治理篇（13–16）

### 第 13 章 Agent 的可观测性

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/04-governance/%E7%AC%AC%2013%20%E7%AB%A0%E3%80%80Agent%20%E7%9A%84%E5%8F%AF%E8%A7%82%E6%B5%8B%E6%80%A7.md)
- **技术主线：** Metrics、Logs、Trace、事件连接 Task→模型→工具→Sandbox→业务系统，兼顾可靠性、延迟、Token、成本与审计；Hook/Daemon/eBPF 各有采集边界。
- **边界与复核：** 观测只能揭示发生什么，不能自行阻止违规；审计需保留真实事件、身份和时序证据。

### 第 14 章 Agent 安全

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/04-governance/%E7%AC%AC%2014%20%E7%AB%A0%E3%80%80Agent%20%E5%AE%89%E5%85%A8.md)
- **技术主线：** 既防 Prompt Injection/外部材料操纵 Agent，也限制 Agent 自主行动影响半径；身份、最小权限、网络出域、Secrets 和审批要跨层布设。
- **边界与复核：** 工具输出是低信任数据；Prompt 中的“不要越权”不等同于执行层强制策略。

### 第 15 章 AI 资产的发现与管理

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/04-governance/%E7%AC%AC%2015%20%E7%AB%A0%E3%80%80AI%20%E8%B5%84%E4%BA%A7%E7%9A%84%E5%8F%91%E7%8E%B0%E4%B8%8E%E7%AE%A1%E7%90%86.md)
- **技术主线：** 注册并版本治理 Prompt、Skill、MCP Server、Agent/AgentSpec；稳态 Agent 可声明依赖，动态 Agent 需按任务发现并筛选能力。
- **边界与复核：** 发现能力≠授权调用，更不表示应把全部工具描述装进上下文。

### 第 16 章 Agent 行为生成与质量验证

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/04-governance/%E7%AC%AC%2016%20%E7%AB%A0%E3%80%80Agent%20%E8%A1%8C%E4%B8%BA%E7%94%9F%E6%88%90%E4%B8%8E%E8%B4%A8%E9%87%8F%E9%AA%8C%E8%AF%81.md)
- **技术主线：** Agent Simulation 以 Scenario Spec/Manifest/Run/Result 组织模拟用户、工具环境和模型回放，形成上线前可重复的行为证据。
- **边界与复核：** Simulation 产生运行证据；Evaluation 才按业务成功准则判断质量，二者不能混为一谈。

## 五、调优篇（17–24）

### 第 17 章 模型调优

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2017%20%E7%AB%A0%E3%80%80%E6%A8%A1%E5%9E%8B%E8%B0%83%E4%BC%98/01%20%E6%A8%A1%E5%9E%8B%E8%B0%83%E4%BC%98%E7%BB%BC%E8%BF%B0.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2017%20%E7%AB%A0%E3%80%80%E6%A8%A1%E5%9E%8B%E8%B0%83%E4%BC%98/02%20%E6%B7%B1%E5%85%A5%E6%B5%85%E5%87%BA%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6.md)
- **技术主线：** 先排除工具、上下文、状态和 Harness 故障，再针对模型能力缺口选择 SFT、Agentic RL 或蒸馏；附篇介绍推理框架吞吐、KV Cache、prefill/decode 等优化。
- **边界与复核：** 推理框架优化不直接修复业务任务逻辑，训练后仍需连同 Harness 验证完整任务。

### 第 18 章 Agent 调优总览和实践

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2018%20%E7%AB%A0%E3%80%80Agent%20%E8%B0%83%E4%BC%98%E6%80%BB%E8%A7%88%E5%92%8C%E5%AE%9E%E8%B7%B5/01%20Agent%20%E8%B0%83%E4%BC%98%E6%80%BB%E8%A7%88.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2018%20%E7%AB%A0%E3%80%80Agent%20%E8%B0%83%E4%BC%98%E6%80%BB%E8%A7%88%E5%92%8C%E5%AE%9E%E8%B7%B5/02%20Agent%20%E6%95%B0%E6%8D%AE%E9%A3%9E%E8%BD%AE%E5%B9%B3%E5%8F%B0%E7%9A%84%E9%80%89%E5%9E%8B%E5%92%8C%E5%AE%9E%E8%B7%B5.md)
- **技术主线：** 以运行轨迹、数据加工、Dataset、评估、实验和 Skill/Harness 候选修改形成数据飞轮；另一篇展示 1688 宙斯接入 AgentLoop 的工程实践。
- **边界与复核：** 真实业务成功与全链路时间、人工审查、模型成本共同评价；个别业务案例指标不能直接外推。

### 第 19 章 Agent 轨迹数据

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2019%20%E7%AB%A0%E3%80%80Agent%20%E8%BD%A8%E8%BF%B9%E6%95%B0%E6%8D%AE.md)
- **技术主线：** 区分单次 Trace、跨轮 Session 和可评价的 Trajectory；检索工具调用及 Observation、终态 Artifact，再抽取有证据的故障样本。
- **边界与复核：** 低分回答不一定是模型错，可能是上下文断裂、工具失败或观测采集缺失。

### 第 20 章 Agent 运行时数据处理

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2020%20%E7%AB%A0%E3%80%80Agent%20%E8%BF%90%E8%A1%8C%E6%97%B6%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md)
- **技术主线：** Pipeline 将真实 Trace 按采样、去重、字段映射、AI 辅助分类和周期任务组织成可复核 Dataset；先预览真实样本再自动化。
- **边界与复核：** 未标记来源与人工核验的合成材料不能直接算黄金样本。

### 第 21 章 Agent 黄金数据集

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2021%20%E7%AB%A0%E3%80%80Agent%20%E9%BB%84%E9%87%91%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- **技术主线：** 黄金任务集包含起始条件、参考结果与经业务确认的 Rubric；候选数据经专家判定后版本化，用于基线与回归。
- **边界与复核：** 调优样本与最终独立验证样本必须区分，避免用熟题证明泛化。

### 第 22 章 Agent 优化：Badcase

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2022%20%E7%AB%A0%E3%80%80Agent%20%E4%BC%98%E5%8C%96%EF%BC%9ABadcase.md)
- **技术主线：** 对历史执行做评价，再让改进后的 Agent 真正重跑任务并比较 Baseline、候选结果与运行成本；Code/LLM/Agent Grader 选择依据任务判定要求。
- **边界与复核：** 评分器本身也要校准；仅重评旧答案不是对修改后 Agent 的实验。

### 第 23 章 受控自进化

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2023%20%E7%AB%A0%E3%80%80%E5%8F%97%E6%8E%A7%E8%87%AA%E8%BF%9B%E5%8C%96.md)
- **技术主线：** 经验召回、Skill、SQL/Script、Workflow 与 Harness 代码分别适合不同问题；来源、适用范围、变更版本和回归验证决定是否采用。
- **边界与复核：** “自进化”并非模型可未经审批修改生产自身；应建立准入、回滚和人工责任边界。

### 第 24 章 Agent 边缘运行时与全球优化

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2024%20%E7%AB%A0%E3%80%80Agent%20%E8%BE%B9%E7%BC%98%E8%BF%90%E8%A1%8C%E6%97%B6%E4%B8%8E%E5%85%A8%E7%90%83%E4%BC%98%E5%8C%96.md)
- **技术主线：** 按 Region 和 Edge 分工优化全球 TTFT/延迟、传输、缓存、安全和设计资产分发；全局基线与地区差异化策略并存。
- **边界与复核：** 边缘只能安全地完成适合缓存的低风险读请求；有一致性、授权与写副作用的任务必须回到权威区域；WebMCP 属早期探索。

## 六、实践篇（25–29）

### 第 25 章 研发效能

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/README.md)
- **技术主线：** 包含 Kitta 专家 Code Review、PatchPilot 内核补丁交付、ABACI 定向测试、STAROps 评测、AgentTeam 研发小队、PolarDB-X 自动修复及组织协作案例；共同强调证据、可复现验证和人的终审。
- **边界与复核：** 厂商/客户提供的 97% 审查准确率、54.1% 目标到达率等数字应注明样本、任务与自述来源，不作为通用 benchmark。

### 第 26 章 设计工程

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2026%20%E7%AB%A0%E3%80%80%E8%AE%BE%E8%AE%A1%E5%B7%A5%E7%A8%8B/GenUI%EF%BC%9A%E8%AE%A9%20Agent%20%E4%BB%8E%E7%BB%99%E5%87%BA%E7%AD%94%E6%A1%88%E8%B5%B0%E5%90%91%E4%BA%A4%E4%BB%98%E7%BB%93%E6%9E%9C.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2026%20%E7%AB%A0%E3%80%80%E8%AE%BE%E8%AE%A1%E5%B7%A5%E7%A8%8B/Vibe%20Designing%EF%BC%9A%E6%84%8F%E5%9B%BE%E9%A9%B1%E5%8A%A8%E7%9A%84AI%E8%AE%BE%E8%AE%A1%E8%8C%83%E5%BC%8F%E8%BF%9B%E5%8C%96.md)
- **技术主线：** GenUI 使用共享组件/Block/Template 生成受约束的 UI Spec，客户端渲染，交互事件回到服务端校验权限和执行；Vibe Designing 将设计资产、动态交互与体验反馈形成闭环。
- **边界与复核：** Schema 能校验结构，不能校验业务数字真伪；界面点“确认”不等于服务端已授权。

### 第 27 章 运维、安全与企业 IT

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E5%90%89%E5%88%A9%E6%B1%BD%E8%BD%A6%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E7%9A%84%E8%90%BD%E5%9C%B0%E5%AE%9E%E8%B7%B5.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E5%A1%94%E6%96%AF%E6%B1%80%E4%B8%87%E5%BA%97%E8%BF%9E%E9%94%81%E7%9A%84%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E9%97%AD%E7%8E%AF%E5%AE%9E%E8%B7%B5.md) · [配套正文 3](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E7%95%85%E6%8D%B7%E9%80%9A%E7%9A%84%E5%8F%AF%E8%A7%82%E6%B5%8B%E4%B8%8E%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E5%AE%9E%E8%B7%B5.md)
- **技术主线：** 吉利、塔斯汀、畅捷通案例强调先建完整可观测与数据源、收敛告警、封装专家 SOP，之后再用 STAROps 辅助故障定位与持续反馈。
- **边界与复核：** “目标 SLA”“未来规划”和已验证上线效果不是同一层级；案例观察不能直接推广为所有行业。

### 第 28 章 客户、销售与运营

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/MiniMax%20%E6%9E%84%E5%BB%BA%E6%B5%B7%E9%87%8F%E9%95%BF%E5%91%A8%E6%9C%9F%E8%AE%B0%E5%BF%86%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7%E7%9A%84%E5%AE%9E%E8%B7%B5.md) · [配套正文 2](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E5%93%94%E5%93%A9%E5%93%94%E5%93%A9%E6%9E%84%E5%BB%BA%E5%85%A8%E5%9F%9F%E5%86%85%E5%AE%B9%E6%B4%9E%E5%AF%9F%E7%9A%84%E5%AE%9E%E8%B7%B5.md) · [配套正文 3](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E4%BC%9A%E8%AE%A1%E5%B8%88%E4%BA%8B%E5%8A%A1%E6%89%80%E4%BF%A1%E6%B0%B8%E4%B8%AD%E5%92%8C%E7%9A%84%E5%8A%9E%E5%85%AC%E6%8F%90%E6%95%88%E6%8E%A2%E7%B4%A2.md) · [配套正文 4](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E4%BB%8E%E5%8D%95%E7%82%B9%E6%8E%A2%E7%B4%A2%E5%88%B0%E8%A7%84%E6%A8%A1%E5%8C%96%E8%BF%90%E8%90%A5%EF%BC%9A%E6%BB%94%E6%90%8F%E4%BD%93%E8%82%B2%E7%9A%84%E4%BC%81%E4%B8%9A%E7%BA%A7%E6%99%BA%E8%83%BD%E4%BD%93%E5%AE%9E%E8%B7%B5.md)
- **技术主线：** MiniMax 长期记忆数据库底座、B 站多模态内容洞察、信永中和多智能体办公服务和滔搏体育零售 Skills 都涉及长期数据、权限与真实业务反馈。
- **边界与复核：** 案例效益来自企业及厂商叙述，指标应独立看口径、规模和环境，不证明模型/Harness 本身的通用性能。

### 第 29 章 GOAI Agent Infra 赛道

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2029%20%E7%AB%A0%E3%80%80GOAI%20Agent%20Infra%20%E8%B5%9B%E9%81%93%EF%BC%9A%E5%A4%9A%20Agent%20%E5%8D%8F%E5%90%8C%E7%9A%84%E5%89%8D%E6%B2%BF%E5%AE%9E%E8%B7%B5%E6%8E%A2%E7%B4%A2.md)
- **技术主线：** 概述 15 支入围团队及九个重点探索：EnergyMesh、FinFlux、总工之眼、店巡、SceneGuard、RevGuard、DataFlow-Agent、CyberGuard、OpsKeeper；强调职责分离、审批、回滚和拒绝执行也是正确终态。
- **边界与复核：** 竞赛演示和架构探索不同于长期生产实绩，不能把参赛选手结论当作可靠性认证。

## 七、总结与展望（30）

### 第 30 章 从 Agentic Application 到 Agentic OS

- **官方原文：** [固定版本章节](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/07-conclusion/%E7%AC%AC%2030%20%E7%AB%A0%E3%80%80%E4%BB%8E%20Agentic%20Application%20%E5%88%B0%20Agentic%20OS.md)
- **技术主线：** 提出可复用、需强制、可独立验证的系统责任下沉；Agentic OS 是九类管理对象与三层职责的架构讨论，不是已成熟的 Linux 替代品。
- **边界与复核：** 接口存在不等于解耦成功；须在不重写任务语义下替换模块，并用同一评估集证实结果等价。

## 第 25–28 章的案例阅读索引

以下按官方章节文件拆分，避免将“案例综述”误当只有四份原文。

- **第 25 章** [智能体：增强 Code Review 的智能体实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E6%99%BA%E8%83%BD%E4%BD%93%EF%BC%9A%E5%A2%9E%E5%BC%BA%20Code%20Review%20%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E5%AE%9E%E8%B7%B5.md)：Kitta：领域专家资产+分级评审+在线/离线修正；维护者承担最后合入责任。
- **第 25 章** [智能体：提升内核补丁交付效率的实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E6%99%BA%E8%83%BD%E4%BD%93%EF%BC%9A%E6%8F%90%E5%8D%87%E5%86%85%E6%A0%B8%E8%A1%A5%E4%B8%81%E4%BA%A4%E4%BB%98%E6%95%88%E7%8E%87%E7%9A%84%E5%AE%9E%E8%B7%B5.md)：PatchPilot：补丁目标/前置依赖→冲突分析→AI 修复→语义回检，无法判断的高风险变更上交人工。
- **第 25 章** [智能体：运维领域的智能体评测和调优实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E6%99%BA%E8%83%BD%E4%BD%93%EF%BC%9A%E8%BF%90%E7%BB%B4%E9%A2%86%E5%9F%9F%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E8%AF%84%E6%B5%8B%E5%92%8C%E8%B0%83%E4%BC%98%E5%AE%9E%E8%B7%B5.md)：STAROps：故障注入评测要求保全告警时间窗、查询证据与因果链，评测样本自身也要持续复核。
- **第 25 章** [智能体：面向内核补丁的定向测试与缺陷检测智能体](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E6%99%BA%E8%83%BD%E4%BD%93%EF%BC%9A%E9%9D%A2%E5%90%91%E5%86%85%E6%A0%B8%E8%A1%A5%E4%B8%81%E7%9A%84%E5%AE%9A%E5%90%91%E6%B5%8B%E8%AF%95%E4%B8%8E%E7%BC%BA%E9%99%B7%E6%A3%80%E6%B5%8B%E6%99%BA%E8%83%BD%E4%BD%93.md)：ABACI：LLM 约束目标调用骨架，运行时方向性反馈迭代，避免用随机覆盖率代替目标代码可达性。
- **第 25 章** [组织协作：Agentic SOC 安全产品⾃进化背后的组织重构](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E7%BB%84%E7%BB%87%E5%8D%8F%E4%BD%9C%EF%BC%9AAgentic%20SOC%20%E5%AE%89%E5%85%A8%E4%BA%A7%E5%93%81%E2%BE%83%E8%BF%9B%E5%8C%96%E8%83%8C%E5%90%8E%E7%9A%84%E7%BB%84%E7%BB%87%E9%87%8D%E6%9E%84.md)：SOCBench：失败样本→共享结构化字段→多模块一致性回归；目标/质量/权限/停止四类自主边界。
- **第 25 章** [组织协作：云通信领域，研发端到端的交付实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E7%BB%84%E7%BB%87%E5%8D%8F%E4%BD%9C%EF%BC%9A%E4%BA%91%E9%80%9A%E4%BF%A1%E9%A2%86%E5%9F%9F%EF%BC%8C%E7%A0%94%E5%8F%91%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%9A%84%E4%BA%A4%E4%BB%98%E5%AE%9E%E8%B7%B5.md)：数字分身与工序契约链，独立交叉质检、先简单任务后渐进扩大权限。
- **第 25 章** [组织协作：多 Agent 组成研发小队，提升交付效率](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E7%BB%84%E7%BB%87%E5%8D%8F%E4%BD%9C%EF%BC%9A%E5%A4%9A%20Agent%20%E7%BB%84%E6%88%90%E7%A0%94%E5%8F%91%E5%B0%8F%E9%98%9F%EF%BC%8C%E6%8F%90%E5%8D%87%E4%BA%A4%E4%BB%98%E6%95%88%E7%8E%87.md)：AgentCore+AgentLoop：不同角色执行和独立 Judge 评审，保留上下文与实验基线。
- **第 25 章** [组织协作：数据库领域，从报警到自动修复的端到端实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2025%20%E7%AB%A0%E3%80%80%E7%A0%94%E5%8F%91%E6%95%88%E8%83%BD/%E7%BB%84%E7%BB%87%E5%8D%8F%E4%BD%9C%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%BA%93%E9%A2%86%E5%9F%9F%EF%BC%8C%E4%BB%8E%E6%8A%A5%E8%AD%A6%E5%88%B0%E8%87%AA%E5%8A%A8%E4%BF%AE%E5%A4%8D%E7%9A%84%E7%AB%AF%E5%88%B0%E7%AB%AF%E5%AE%9E%E8%B7%B5.md)：PolarDB-X：报警→稳定复现→修复前后对照测试→回归→人工代码审查。
- **第 26 章** [GenUI：让 Agent 从给出答案走向交付结果](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2026%20%E7%AB%A0%E3%80%80%E8%AE%BE%E8%AE%A1%E5%B7%A5%E7%A8%8B/GenUI%EF%BC%9A%E8%AE%A9%20Agent%20%E4%BB%8E%E7%BB%99%E5%87%BA%E7%AD%94%E6%A1%88%E8%B5%B0%E5%90%91%E4%BA%A4%E4%BB%98%E7%BB%93%E6%9E%9C.md)：GenUI：数据事实与界面 Spec 分开；Server 执行授权，不交给 Client。
- **第 26 章** [Vibe Designing：意图驱动的AI设计范式进化](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2026%20%E7%AB%A0%E3%80%80%E8%AE%BE%E8%AE%A1%E5%B7%A5%E7%A8%8B/Vibe%20Designing%EF%BC%9A%E6%84%8F%E5%9B%BE%E9%A9%B1%E5%8A%A8%E7%9A%84AI%E8%AE%BE%E8%AE%A1%E8%8C%83%E5%BC%8F%E8%BF%9B%E5%8C%96.md)：设计工程资产、动态交互和体验自我校验共同构成设计能力系统。
- **第 27 章** [吉利汽车智能运维的落地实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E5%90%89%E5%88%A9%E6%B1%BD%E8%BD%A6%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E7%9A%84%E8%90%BD%E5%9C%B0%E5%AE%9E%E8%B7%B5.md)：运维数据和历史治理先行，STAROps 诊断与组织转型后置。
- **第 27 章** [塔斯汀万店连锁的智能运维闭环实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E5%A1%94%E6%96%AF%E6%B1%80%E4%B8%87%E5%BA%97%E8%BF%9E%E9%94%81%E7%9A%84%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E9%97%AD%E7%8E%AF%E5%AE%9E%E8%B7%B5.md)：CMS 2.0 归并四路告警、反馈校正 RCA、专家操作转成 Skill 与 Mission。
- **第 27 章** [畅捷通的可观测与智能运维实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2027%20%E7%AB%A0%E3%80%80%E8%BF%90%E7%BB%B4%E3%80%81%E5%AE%89%E5%85%A8%E4%B8%8E%E4%BC%81%E4%B8%9AIT/%E7%95%85%E6%8D%B7%E9%80%9A%E7%9A%84%E5%8F%AF%E8%A7%82%E6%B5%8B%E4%B8%8E%E6%99%BA%E8%83%BD%E8%BF%90%E7%BB%B4%E5%AE%9E%E8%B7%B5.md)：五层可观测→自动化→业务预防/应急/成本闭环；SLA 目标不能写成达成。
- **第 28 章** [MiniMax 构建海量长周期记忆数据底座的实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/MiniMax%20%E6%9E%84%E5%BB%BA%E6%B5%B7%E9%87%8F%E9%95%BF%E5%91%A8%E6%9C%9F%E8%AE%B0%E5%BF%86%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7%E7%9A%84%E5%AE%9E%E8%B7%B5.md)：PolarDB 多主、冷热数据、记忆对象的数据成本与响应延迟取舍。
- **第 28 章** [哔哩哔哩构建全域内容洞察的实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E5%93%94%E5%93%A9%E5%93%94%E5%93%A9%E6%9E%84%E5%BB%BA%E5%85%A8%E5%9F%9F%E5%86%85%E5%AE%B9%E6%B4%9E%E5%AF%9F%E7%9A%84%E5%AE%9E%E8%B7%B5.md)：多模态提取、品牌属性结构化与小模型低成本处理公开去标识化内容。
- **第 28 章** [会计师事务所信永中和的办公提效探索](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E4%BC%9A%E8%AE%A1%E5%B8%88%E4%BA%8B%E5%8A%A1%E6%89%80%E4%BF%A1%E6%B0%B8%E4%B8%AD%E5%92%8C%E7%9A%84%E5%8A%9E%E5%85%AC%E6%8F%90%E6%95%88%E6%8E%A2%E7%B4%A2.md)：AgentCore+AI 网关统一多 Agent 入口、权限与审计。
- **第 28 章** [从单点探索到规模化运营：滔搏体育的企业级智能体实践](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study/%E7%AC%AC%2028%20%E7%AB%A0%E3%80%80%E5%AE%A2%E6%88%B7%E3%80%81%E9%94%80%E5%94%AE%E4%B8%8E%E8%BF%90%E8%90%A5/%E4%BB%8E%E5%8D%95%E7%82%B9%E6%8E%A2%E7%B4%A2%E5%88%B0%E8%A7%84%E6%A8%A1%E5%8C%96%E8%BF%90%E8%90%A5%EF%BC%9A%E6%BB%94%E6%90%8F%E4%BD%93%E8%82%B2%E7%9A%84%E4%BC%81%E4%B8%9A%E7%BA%A7%E6%99%BA%E8%83%BD%E4%BD%93%E5%AE%9E%E8%B7%B5.md)：以零售业务 Skills、各角色 Agent、AgentTeam 与业务结果持续运营，而非只追求 Agent 数量。

## 章节之外的资料（需与 30 章区分）

- [2026 Agent 开发者调研报告](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/2026-agent-survey-report.md)：来自官方 1906 份有效问卷，按受访样本解读生产化、架构、MCP、网关、评估；属于**特定调查样本**，并非全体企业统计。
- **Jev 专题（5 篇）：** 结构化决策模型、AgentScope Java 接入、工单分诊、HiRoute 智能路由和 Judge 评测对比；区分分类/程度/阶段胜任评分与完整任务验收。
- **Personal Agent 专题（3 篇）：** 个人 Agent 的产品趋势、AgentScope Agent-as-a-Service、DSH Super Agent 与 Flock 探索；Task/Ticket 生命周期与 Session 解耦是共性。
- **FDE 专题（1 篇）：** AI Coding 企业项目的 Spec→Design→Tasks 三产物、三阶段三闸门与测试审计；案例数字来自已脱敏的单项目叙述。

## 交叉结论（只反映原书，不代替外部独立验证）

1. **任务对象先于模型对象。** Task/Run/Outcome/Approval 必须有持久语义，模型会话只是执行的一种投影。
2. **探索交给模型，强制边界交给系统。** Tool Schema、MCP、Sandbox、身份与审批不是同一个控制点；真实副作用需独立验证。
3. **运行不等于完成。** Event、Trace、Artifact、业务 Outcome 和评估分别提供不同证据，模型自报结束不是验收依据。
4. **优先修复 Harness 及数据条件。** 只有分离环境、工具、Context 故障后，SFT/RL/蒸馏的成本才可能合理。
5. **调优变更不能绕过发布门禁。** 新 Skill、Prompt、模型或 Harness 修改需要独立黄金数据集、回归、安全测试、审查和可回滚版本。
6. **数字只是具体场景的证据。** 书中的客户案例、竞赛结果与阿里云产品能力不等同于可重复的跨组织基准；需要保留版本、样本、条件、信任边界。

## 阅读边界

本记录来自上游固定版本的官方 Markdown 章节和专题材料，含结构、关键机制、例证与各章结束判断；不声称已经独立运行其中的云产品、复现 Kitta/ABACI/STAROps 等内部实验、确认所有链接、图片与数字，或替作者补完隐含的接口规范。后续上游每周增补时应按 Git diff 重新核对内容与来源。
