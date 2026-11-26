---
type: entity
project_id: memt-1-0
tags:
  - paper
  - llm
  - memory
  - long-context
  - embodied-agent
  - agentic-coding
  - nvidia
status: complete
updated: 2026-11-26
arxiv: "2610.26825"
project: https://github.com/MemTensor/MemT-1.0
code: https://github.com/MemTensor/MemT-1.0
related:
  - ./paper-memcompiler-state-conditioned-memory.md
  - ./paper-spatial-memory-agent.md
  - ../concepts/agentic-coding-software-fundamentals.md
original:
  - ../../sources/papers/memt_1_0_arxiv_2610_26825.md
sources:
  - ../../sources/papers/memt_1_0_arxiv_2610_26825.md
  - ../../sources/sites/memt-1-0.md
  - ../../sources/repos/memtensor-memt-1-0.md
summary: "MemT-1.0 将持续演化的局部记忆与可训练超记忆纳入模型计算；Pro/Flash/Nano 在论文所比较的参数化记忆方法中表现领先，但文本记忆系统在部分长历史证据检索任务上仍有优势，机器人应用需单独验证。"
---

# MemT-1.0: Towards Memory-Native Evolution（面向原生记忆演化的记忆基础模型）

## 一句话定义

**MemT-1.0** 是一组将持久、动态更新的记忆纳入模型计算的语言模型；它在生成过程中读取内部记忆，并在轮次或片段结束时更新记忆状态，从而减少对原始长上下文的持续访问。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GDN-2 | Gated DeltaNet-2 | 论文用作轮次或片段结束时稳定的逐 token 记忆更新机制 |
| MoS | Mixture-of-Storage | 用于扩展原生记忆容量的存储混合机制 |
| KV cache | Key-Value Cache | Transformer 解码时保留历史键值的缓存，论文以完整上下文 KV cache 作存储比较 |
| SGLang | Structured Generation Language | 官方推理集成提供的模型服务框架 |

## 为什么重要

- **记忆变成模型状态的一部分：** 常见 RAG 或外置摘要会在生成前检索、拼接文本；MemT-1.0 将可演化的记忆状态纳入模型计算，生成时使用记忆注意力。
- **把历史长度与持久状态大小部分解耦：** 作者报告在 128K 历史实验中，Nano 相较完整上下文 KV cache 的持久存储约减少 66 倍；这属于论文给定配置下的比较，不代表无限容量或所有硬件上的吞吐保证。
- **面向程序性经验复用：** 论文还展示 coding agent 在长程任务中重用经验，不必把经验文本持续追加到任务上下文。该结果提示可研究机器人任务层的程序性记忆，但尚不能推导出机器人部署效果。

## 核心原理

MemT-1.0 在 backbone attention 旁加入 Native Memory Block：

| 组件 | 职责 |
|------|------|
| Local Memory Block | 维护随交互变化的多通道动态记忆矩阵 |
| Hyper Memory Block | 通过可训练的记忆 query/key/value 投影与 erase/write/forget/output gates 调节记忆读写 |
| Memory attention | 生成时读取当前状态，对读出归一化、门控后并入 backbone attention 输出 |
| GDN-2 更新 | 在轮次或片段结束时逐 token 更新记忆，供后续生成使用 |

Mixture-of-Storage（MoS）进一步用于扩展原生记忆容量。训练路线包括 backbone preparation、continued pre-training 与 post-training；作者称训练数据规模为 34B tokens。

### 流程总览

```mermaid
flowchart LR
  history["历史交互"] --> update["GDN-2 逐 token 更新"]
  update --> local["动态 Local Memory"]
  hyper["可训练 Hyper Memory"] --> read["Memory attention 读取"]
  local --> read
  prompt["当前输入"] --> backbone["Backbone attention"]
  read --> combine["门控记忆读出与 backbone 合并"]
  backbone --> combine
  combine --> response["生成响应"]
  response --> history
```

## 模型族与评测

| 模型 | Backbone | 总参数 | 激活参数 | 可训练记忆参数 | 动态记忆状态 |
|------|----------|--------|----------|----------------|--------------|
| MemT-1.0-Nano | Qwen3.6-35B-A3B | 36B | 3B | 451M | 10M |
| MemT-1.0-Flash | DeepSeek-V4-Flash | 288B | 17B | 3.61B | 336M |
| MemT-1.0-Pro | NVIDIA-Nemotron-3-Ultra-550B-A55B | 554B | 59B | 4.23B | 13M |

报告评估 11 项记忆基准。作者报告 Pro、Flash、Nano 在被比较的**参数化记忆方法**中依次排名前三；强文本记忆系统整体仍领先，尤其是长历史中需要细粒度证据检索的任务。论文还报告 Nano 在 128K 历史长度下相对完整上下文 KV cache 约 1.86×端到端推理加速、持久存储减少约 66×。§6.5 图 7 的端到端时间包含历史准备与固定 128 token 生成，查询 TTFT 排除历史准备（但包含状态恢复、查询预填充）；128K 的查询 TTFT 加速约 2.44×，不是端到端的 1.86×。测量使用 HF 推理、两张 GPU、并发与 batch 均为 1，不是下方 SGLang Pro 部署压测。

### v5 表 5：Nano 架构与训练消融

v5 将表 5 的显示精度提高到两位小数。Nano 基线为 46.97；将逐 token 更新改为逐轮聚合后为 40.98（−5.99 分），去掉稳定化机制为 46.43（−0.54），换为 GDN 为 46.72（−0.25）。去掉持续预训练（CPT）为 44.85，去掉后训练（PT）为 36.28。该实验中写入粒度的影响大于上述更新规则替换，后训练尤其重要。这是 Nano 的消融，不是 Pro 的得分。

### Pro 训练与通用能力

v5（2026-11-23）§3.6：Pro 冻结 Nemotron 骨干，只训练记忆模块；CPT 使用 CE，后训练以全上下文骨干教师的 Top-64 目标计算 KL 损失。§6.3 表 4 的噪声记忆设置下，Pro 相对同设置骨干在六项通用基准上平均降低 0.11 个百分点，LongBench v2 降低 5.17 个百分点。不可把平均接近无损理解为各任务无损，也不可将 Nano 的 128K 效率与 SWE-bench 结果转归 Pro。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant App as Caller
  participant Session as MemTSession.chat
  participant Client as NativeClient
  participant Server as SGLang service
  participant State as Session memory
  App->>Session: chat（按需打开自动写入）
  Session->>Client: lease and query
  Client->>Server: acquire and POST /generate query
  Server->>State: Read committed state
  State-->>Server: Memory readout
  Server-->>Client: Generated token IDs
  Client-->>Session: Generation result
  alt 轮次正常结束且开启自动写入
    Session->>Client: Replay unsaved turns via commit_turns
    Client->>Server: begin and POST /generate commit per turn
    Server->>State: Update candidate memory
    Client->>Server: publish
    Session->>Session: publish 成功后清除已提交轮次
  else 普通聊天、输出截断或等待工具结果
    Session->>Session: 保留未提交上下文，不自动写入
  end
  Client->>Server: release lease
  Session-->>App: Answer
```

读记忆默认开启，`enable_memory` 控制的是自动写入，默认关闭，两者不是同一个开关。时序依据 `inference/memt_sglang/client/session.py` 的 `chat`、`_generate`、`_commit_turns` 和官方 inference guide 的 HTTP 请求表；工具调用未结束或生成截断时不自动提交。Pro 服务入口为 `inference/start_memt_pro.sh` → `python -m memt_sglang.launch`。快照保存使用 `save_state` 提交历史后 export/publish，恢复使用 `load_state` import/publish。完整步骤见 [官方仓库](../../sources/repos/memtensor-memt-1-0.md)。

## 工程实践

- **先区分应用层级：** MemT-1.0 的论文应用是长程 coding agent。迁移到机器人时，可把它视为任务级历史 / 程序性记忆候选；不要把它当作 100–1000 Hz 的运动控制器或实时安全模块。
- **核对部署资源：** 官方 Nano 示例以两张 GPU 启动 SGLang 服务；Pro 官方启动示例为 8 GPU、TP8/EP8/DP1、BF16、16384 上下文；实际设备需容纳完整权重、运行态及缓存。Flash 和 Pro 的参数量远大于 Nano。实际显存、吞吐、并发和量化效果应在目标硬件与模型发行版本上测量。
- **拆分许可：** 仓库推理软件使用 PolyForm Noncommercial License 1.0.0，论文材料使用 CC BY-NC-SA 4.0；Hugging Face Flash 模型卡标注权重为 MIT。Pro 的 Hugging Face 模型卡为 OpenMDW 1.1，ModelScope 入口由官方 README 列出但本次未能直接核查。第三方 backbone、组件及其他权重版本还需遵守其各自条款。
- **做任务级对照：** 与完整上下文、RAG 或外置摘要比较时，分别记录长程任务成功率、事实检索准确性、持久状态大小、端到端延迟和显存；特别检查需要引用早期细节的任务，因为报告承认强文本记忆在此类任务上仍具优势。

## 局限与风险

- **有限容量与证据检索：** 固定大小的动态状态不能保证无损保留历史细节。v5 表 3 的记忆基准平均分为 Pro 53.08、Flash 50.60、Nano 46.97，而文本记忆系统 MemOS NV 为 72.08；精确证据回溯仍需对照检索方案。
- **容量扩展不等于发行模型实测：** §5 的多专家实验展示 MoS 扩展趋势，但附录 B 表 7 的三个主评测模型均采用单专家；不能将多专家消融成绩直接归于发布模型。
- **效率边界：** §6 的 128K 对比属于 Nano，较短历史下未必比完整上下文更快；逻辑记忆状态的容量不包含模型权重和当前 KV cache，不能等同于部署总显存。
- **持续学习边界：** §7 的程序性经验实验依赖外部收集与写入的经验轨迹；§8 仍讨论有限状态与长期记忆演化的限制，未证明自主将局部记忆整合到可训练超记忆的闭环。机器人任务成功率、安全性及实时性需要独立验证。

## 与其他工作对比

| 路线 | 记忆所在位置 | 与 MemT-1.0 的差异 |
|------|--------------|--------------------|
| 完整上下文（KV cache） | 推理时保留全部历史 token | 论文在 128K 历史下报告 Nano 持久存储约减少 66×、端到端约 1.86× 加速；代价是历史被压缩进固定状态 |
| RAG / 外置摘要（文本记忆系统） | 模型外部的可检索文本 | 生成前检索拼接；论文承认强文本记忆在需细粒度证据检索的长历史任务上整体仍领先 |
| 其他参数化记忆方法 | 模型内可训练参数 / 状态 | 同类对照；作者报告 Pro、Flash、Nano 在被比较方法中依次排名前三 |
| [MemCompiler](./paper-memcompiler-state-conditioned-memory.md) | 按状态条件编译的外部经验 | 面向具身智能体逐步编译经验；MemT-1.0 则把记忆写进模型内部状态 |

## 结论

**MemT-1.0 展示了把可更新记忆作为模型内部状态的一条可扩展路径，优势主要落在参数化记忆与长程经验复用；它没有证明原生记忆普遍优于检索式文本记忆。**

- 若任务需要长期复用程序性经验，可将 MemT-1.0 纳入候选并与外置 RAG / 摘要对照。
- 读取 128K 效率数字时，保留 Nano、指定 KV cache 基线和论文实验配置，不外推成通用性能承诺。
- 对需要精确回忆长历史细节的任务，优先单独测事实证据检索，因为论文承认文本记忆系统整体仍领先。
- 将其迁移到具身系统时，限制在任务层记忆研究；实时运动控制和安全闭环仍需独立设计与验证。
- 商业部署前分别检查仓库软件许可、目标权重许可和 backbone / 第三方组件许可。

## 关联页面

- [模型架构纵深路线](../../roadmap/depth-model-architecture.md) — 原生记忆作为注意力并行支路的骨干架构案例；非机器人策略验证。

- [Coding agent 验证闭环](../methods/coding-agent-verification-loops.md) — 将经验注入与实际执行验证分开，避免把记住一条经验当成它已被正确执行。
- [Agentic manipulation policies](../methods/agentic-manipulation-policies.md) — 任务级历史与机器人策略接口的对照，不代表已有真机迁移。
- [MemCompiler：具身智能体的状态条件记忆编译](./paper-memcompiler-state-conditioned-memory.md) — 对照逐步编译外部经验与模型内部原生记忆。
- [Spatial Memory Agent](./paper-spatial-memory-agent.md) — 比较具身任务中的空间/程序性经验组织方式。
- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 将长程 coding agent 应用放进 agent 工程背景理解。

## 参考来源

- [MemT-1.0 技术报告摘录](../../sources/papers/memt_1_0_arxiv_2610_26825.md)
- [官方项目总览](../../sources/sites/memt-1-0.md)
- [官方实现仓库](../../sources/repos/memtensor-memt-1-0.md)

## 推荐继续阅读

- [Hugging Face Pro](https://huggingface.co/MemTensor/MemT-1.0-Pro) · [ModelScope Pro](https://www.modelscope.cn/models/MemTensor/MemT-1.0-Pro) — Pro 权重发布入口。
- [技术报告 v5 HTML](https://arxiv.org/html/2610.26825v5) · [PDF](https://arxiv.org/pdf/2610.26825v5) — 本次全文及技术附录核对版本。

- [MemT-1.0 technical report (arXiv:2610.26825)](https://arxiv.org/abs/2610.26825) — 模型结构、训练、基准与资源效率实验。
- [Official inference repository](https://github.com/MemTensor/MemT-1.0) — 推理代码、SGLang 集成与部署说明。
