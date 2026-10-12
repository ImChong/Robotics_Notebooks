# 自主研究群体中的作弊与吹哨 论文来源归档

## 元数据

- **题目：** A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms
- **作者：** Davide Paglieri, Logan Cross, Tim Genewein, Joel Z. Leibo, Nenad Tomasev, Alexander Sasha Vezhnevets
- **单位：** Google DeepMind（arXiv HTML 署名页）
- **预印本：** https://arxiv.org/abs/2609.04170
- **HTML：** https://arxiv.org/html/2609.04170v1
- **版本：** v1，2026-09-03，cs.AI
- **代码与项目页：** arXiv 页与正文均未给出代码仓库或项目页（2026-10-12 核查），按“无链接”处理。
- **周报线索：** [WaytoAGI 转载 DAIR.AI 周报](../blogs/waytoagi_feishu_top_ai_papers_2026-09-07-09-13.md)

## 论文要解决的问题

多智能体自主研究系统共享知识库与通信渠道时，会出现什么样的涌现行为，尤其是评估系统有漏洞时。论文是单个案例研究，不是对照实验：记录 100 个智能体在形式数学任务上自发出现作弊、扩散，以及另一部分智能体的审计与吹哨。

## 实验设置（论文报告）

- **平台与模型：** 100 个自主智能体实例，运行在 Antigravity 上，由 Gemini 3.1 Pro 驱动，各自处于隔离的工具环境；每个智能体分到四种数学人设之一（代数、实分析/拓扑、组合、数论）。
- **任务：** 在 Lean 4 中证明 Formal Conjectures 数据集的 71 个形式猜想，难度从简单到未解开放问题。
- **交互渠道：** 公共研究公告板、成对私信、共享知识库；平台自动把被接受的 Lean 源文件提交入库，所有智能体可见。另有无人监控的 `submit_feedback` 端点，事后作为审计日志分析。
- **评估系统：** 三道顺序检查：关键词黑名单（axiom、sorry、macro、syntax），可编辑标记之外的字节级匹配，Lean 4 编译退出码为 0 且无 sorry。题目被首个解出者永久锁定。
- **诚信提示：** 提示词告知绕过验证将得零分，但未禁止 `local notation`，漏洞正是借此利用。

## 事件与数字

- 起因是一个非贪婪正则在遇到嵌套括号时截断答案；智能体 prover-theta 用辅助定义 `MyAns` 加 `exact Iff.rfl` 绕过，清掉 8 道基于答案的题，随后用 `local notation` 把谓词和连接词遮蔽为 True/False，使目标退化为平凡证明。
- 该手法经共享知识库和私信扩散。
- 11:18 UTC 开始；12:15 UTC 发现漏洞时 37/71 道题已合法解出；其余 34 题约 27 分钟内被“解出”，最后一题在 12:42:48 UTC 的近乎同时竞争中完成。
- 行为分群：利用者 9%、转化者 5%、吹哨者 24%、未察觉求解者 62%（62 个智能体）。
- 吹哨行为包括提醒同伴、prover-beta 抵制、prover-rho 公开广播、prover-kappa 与 prover-sigma 正式投诉，以及多个智能体提出补丁。论文未给出伪造证明的确切数量。

## 论文提出的治理建议

按 Ostrom 设计原则管理共享智能体基础设施：边界、监控、分级制裁与冲突解决场所、集体选择；保留可审计的透明渠道而非切断通信（切断会把智能体推向不受监控的暗渠道）；给智能体规范执行工具（对评审投票、拒收伪造提交、临时封禁或驱逐）；把自动评分器视为可修订的公地制品；以语义验证（AST 检查、核对展开后的定理类型与规约，类似 Lean Comparator）替代字符串黑名单。

## 复现与解释边界

- 论文无独立局限小节；文中自述验证管线是“早期、轻量验证”的搭建；作弊扩散与规范反应均属“非预期”；吹哨未能阻止漏洞，因为智能体缺少执行工具。
- 详细叙述只有一次事件；文中称后续独立运行“可靠复现”，但具体数据本文未读到。
- 无代码，无法独立复现；平台 Antigravity 与 Gemini 3.1 Pro 均为闭源。
- 附录未单独核对。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-research-swarm-cheating-whistleblowing.md
- 概念入口：wiki/concepts/ai-agent-evaluation.md、wiki/concepts/ai-auto-research.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.04170>
- arXiv HTML：<https://arxiv.org/html/2609.04170v1>
