# Using a "chaos loop" to harden the Claude Docs launch / 用持续破坏性测试闭环加固 Claude Docs 服务

> 一手来源归档：Anthropic Claude Code 官方工程博客的结构化研读记录；非逐字转载。

- **类型：** blog（engineering）
- **作者：** Vivek Nair
- **机构：** Anthropic
- **发布日期：** 2026-11-17
- **原文：** https://claude.dev/blog/chaos-loop-claude-docs-launch/
- **版本：** 公开网页发布版，2026-11-19 核对
- **核对范围：** 通读网页 The chaos loop、Making the service legible to Claude、Launch week、Why this works now、What it cost、Trying this out 全部正文；核对发布规模、内存优化、人工门禁、技术选型与成本的上下文。
- **通读：** 已完成
- **沉淀到 wiki：** [Agent 驱动的混沌压测与修复闭环](../../wiki/methods/agent-driven-chaos-testing-loop.md)
- **源码可用性：** 文章没有提供用于该 Docs 服务和内部 chaos harness 的公开可运行源码仓库，本文无法据此验证生产实现或绘制源码运行时序图。

## 背景与目标

Claude Docs 2026-09-16 对外上线。上线前的预生产系统最初只有单个节点、几千内部用户，目标却是处理上线初期百万量级的文档。作者在 9 月 4 日左右让 Claude Code 协调 staging 的压力与故障注入：选场景 → 跑 harness → 读可观测数据 → 生成修复建议 → 对可逆变更实施并部署 → 重测。约连续运行两周；协调会话委派给多个 sub-agent，在繁忙一周通常同时约 11 个、峰值 30 个。

**不变量：** 文档（包括长编辑历史）必须几乎即时打开；服务端确认后的编辑不能丢失。不可轻易回滚的数据库 schema、key layout、文档存储/物化方案，需要人先批准。作者明确将人工选择与最终批准放在反馈链中，并非全自动上线生产。

## 原文给出的压测场景和诊断

- 一天创建 1000 万文档；2.2 万实时连接同时重连；数百观众访问同一热门文档；加载文档时杀掉服务器；数据库停滞。
- Harness 在 staging 饱和施压，Claude 通过 Datadog MCP 读取遥测，给出选项和建议；细化了数据库 query 名称、扫描/返回行数等可归因信号。
- 初期某大型文档在 pod 内占用约 3GB。第一次缩小缓存的修复被后续实验推翻；细分内存 profiling 找到每次按键多次复制数据结构，改用 copy-on-write 后约三天将峰值降至 180MB。
- 作者报告预计所需内存优化型 pods 从 500 多缩至约 50；使用 Rust 与少依赖以提升内存可解释性。作者**未做 Rust 与 Go 的对照试验**，也未证明 GC 不适用所有同类场景。

## 上线结果与成本：作者报告，非独立测量

| 项 | 官方博客报告 |
| --- | --- |
| 服务规模 | 2026-09-22 前创建 460 万余份文档 |
| 请求/错误 | 约 20 亿请求，60 次服务端错误 |
| 发布 | 2026 年 9 月生产环境发布 629 次，无回滚 |
| 事前故障预演 | 预演的 10 类场景中有 7 类在生产真实出现 |
| 推理成本 | 2026 年 8 月末至上线按 Fable 5.2 API 牌价约 12.5 万美元 |
| 小规模类比 | 两周约 3000–10000 美元为作者**未经实际运行的粗估**，不是实验结果 |

## 可复用工程原则与限制

1. 把成功条件写成独立可检查的不变量；不要只说“提高稳定性”。
2. 不断增加观测信号，令每次故障都能定位到系统自身的代码、依赖和数据结构。
3. 让 staging 承接 10 倍量级负载、节点宕机、数据库迟滞、发布中断等失败注入；修复后必须在相同条件重测。
4. 可逆变更由 agent 验证实施，高风险的存储/Schema 变化由人批准；production rollout 仍需独立门禁。
5. 作者使用内部、能力约等于 Claude Fable 5.2 的模型；此案例不能直接推广为普通账户可用的长时间自运行能力或普遍成本。
6. 文章没有提供原始实验日志、对照组、运行脚本或完整公共代码；量化成绩属于作者对生产环境的陈述，不等于本知识库复现。

## 关联资料

- [Claude Code 动态工作流官方原文归档](claude_dev_dynamic_workflows_2026-06-02.md)
- [Coding Agent 验证闭环](../../wiki/methods/coding-agent-verification-loops.md)
- [可观测性：日志、指标与链路追踪](../../wiki/concepts/observability-logs-metrics-tracing.md)

## 原文入口

- [Using a "chaos loop" to harden the Claude Docs launch（2026-11-17）](https://claude.dev/blog/chaos-loop-claude-docs-launch/)
