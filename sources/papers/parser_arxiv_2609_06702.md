# PARSER 论文来源归档

## 元数据

- **题目：** PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents（HTML 页题名写作 ParSer）
- **作者：** Kun Li, Zexuan Qiu, Tianhua Zhang, Irwin King, Helen Meng（Li、Qiu、Zhang 等贡献）
- **单位：** 香港中文大学（The Chinese University of Hong Kong）
- **预印本：** https://arxiv.org/abs/2609.06702（cs.CL；v1 2026-09-06，v2 2026-09-26）
- **HTML：** https://arxiv.org/html/2609.06702v2（长约 17.8 万字符，已读到附录 F；附录 B.1 前后曾被截断，续读后补全）
- **项目页：** https://cuhk-parser.github.io/
- **代码：** https://github.com/cuhk-parser/PARSER（MIT 许可；已核对，见 [repos 归档](../repos/parser-long-context-agents.md)）
- **站点归档：** [sites 归档](../sites/parser-long-context-agents.md)
- **入库日期：** 2026-10-12

## 论文要解决的问题

顺序记忆智能体（MemAgent、ReMemR1 类）一次读一个 chunk 并更新紧凑记忆。这把推理深度绑定在阅读进度上，使准确率对证据位置敏感，延迟随文档长度线性增长。PARSER 把“阅读”和“推理”解耦。

## 方法与训练信息

1. **架构：** 主导智能体只看到问题，以 ReAct 风格循环做多轮 scatter-gather；每个冻结子智能体绑定一个 chunk，并行读取后返回证据或弃权。每轮主导智能体向所有子智能体广播查询、聚合证据、写出更深的后续查询。
2. **模型：** 主导智能体 Qwen3.5-4B / 9B（RL 训练）；子智能体 Qwen3.5-4B 非思考模式（温度 0.7，生成上限 512 token，JSON 输出），亦测试了 2B/9B 子智能体而无需重训。
3. **分块：** 训练 512 token/块，推理 4,096 token/块。
4. **轮数：** 训练总轮数上限 9（每轮 ≤2,048 token），推理上限 12，延迟分析里约 4 轮。
5. **RL：** GRPO，二值 exact-match 奖励，无格式奖励，观测 token 不计入策略梯度。学习率 1e-6、mini-batch 128、15 步 warm-up、clip ε=0.2、KL β=1e-3、组大小 5。
6. **数据：** 从 41,027 道 HotpotQA 题（过滤 Qwen3.5-9B 无文档即可答对者）得 32,768 条 RL 样本，每条 200 段（约 28K token），沿用 MemAgent 配方合成。
7. **框架与算力：** VERL + Megatron 后端、SGLang 推理、全异步 RL；6 张 H100（4 rollout + 2 actor）加 10 张 H100 服务子智能体；4B 训练 180 步约 312 小时，9B 约 400 小时。子智能体 KV 缓存经 SGLang Radix Cache 复用，cache-aware router 把同一 chunk 请求固定到同一实例。

## 论文报告的结果

- **基准：** HotpotQA（域内）与 2WikiMultiHopQA（域外），50–6,400 段（约 7K–896K token），指标 Sub_EM（双向子串匹配，沿用 MemAgent/ReMemR1 评测代码），3 次均值。
- **HotpotQA 平均：** PARSER-4B 84.57，PARSER-9B 86.79；MemAgent-4B 78.58，ReMemR1-4B 78.91，MemAgent-9B 80.05，ReMemR1-9B 78.39；DeepSeek-V4-Pro think-max 80.47，non-think 75.10；Qwen3.5-4B 全上下文思考 65.82（896K 处仅 31.25）。
- **差距：** 4B 比最强顺序基线平均高 5.7 点、896K 高 12.0 点；9B 比 DeepSeek-V4-Pro think-max 高 6.3 点。
- **2Wiki 平均：** PARSER-4B 87.04，PARSER-9B 88.48，DeepSeek-V4-Pro think-max 88.18。50 段时 ReMemR1-4B（88.54）与 4B 全上下文思考（88.28）高于 PARSER，“稳定领先”仅限长子集（≥1,600 段）。
- **鲁棒性（各 512 题）：** 证据位置、顺序（正序/逆序）、证据间距离；MemAgent/ReMemR1 明显波动，PARSER 基本平稳。
- **延迟（896K，摊销秒/样本，Table 2）：** 并发 1：PARSER 78.22 对 MemAgent 876.20（约 11.2 倍）；并发 16：58.86 对 101.94（约 1.7 倍）；并发 32：58.76 对 74.96（约 1.3 倍）。全上下文基线在并发 16/32 因显存无法跑完 896K。作者在引言里对该数字的表述含糊，以表为准。
- **消融：** 子智能体 2B/4B/9B 平均 78.26/84.57/84.77；分块 4,096 为 84.57，整文档 73.76，131,072 为 80.34；RL 前后（HotpotQA 4B）74.42 → 84.57，9B 76.24 → 86.79。
- **RULER（附录 B.2）：** 10 个合成任务、8K–512K，只有热力图，无文字数字。
- **失败案例（附录 F.2）：** 子智能体误读同名人物 chunk，主导智能体采信了错误答案（上下文隔离，无法核对来源）。

## 复现与解释边界

- 训练与延迟实验需 16 张 H100 量级集群；评测可用发布的 4B 检查点。
- 训练数据为 HotpotQA 合成，评测集中在多跳 QA 与 RULER，未覆盖开放式长文档任务。
- 未读到独立 limitations 章节。
- 论文 9.47% 的 DeepSeek-V4-Pro think-max 格式违规说明基线对比受提示协议影响。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-parser-parallel-read-deep-reason.md
- 方法入口：wiki/methods/grpo.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.06702>
- arXiv HTML：<https://arxiv.org/html/2609.06702v2>
- 项目页：<https://cuhk-parser.github.io/>
- 代码：<https://github.com/cuhk-parser/PARSER>
