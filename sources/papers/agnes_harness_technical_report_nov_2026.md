# Agnes Harness Technical Report（2026 年 11 月）

> 来源归档，对应 [独立论文详情](../../wiki/entities/paper-agnes-harness-rsi-technical-report.md)。

- **标题：** Agnes Harness Technical Report: A Ledger-Authoritative, Dual-System & Self-Improving Runtime for Production Agents
- **发布方：** Agnes AI Systems Team；2026-11（官方摘要称 PDF 共 21 页）
- **论文 PDF：** https://github.com/AgnesAI-Labs/Agnes-Harness-Technical-Report-Nov-2026/blob/main/PAPER.pdf
- **官方可读摘要：** https://github.com/AgnesAI-Labs/Agnes-Harness-Technical-Report-Nov-2026/blob/main/REPORT.md
- **实现仓库：** https://github.com/AgnesAI-Labs/agnes-harness
- **核对范围：** 已通读官方 REPORT.md 全文、交叉核验官方 AGH README 与架构及启动入口；**未能读取 PDF 原文件**，不能据此声称全文精读、图表逐项核验或独立复现。
- **通读：** 未完成
- **未完成范围：** 原始 PDF 尚未通读；本条仅核对了作者公开 REPORT.md。
- **入库：** 2026-11-20

## 关键机制

1. **账本与副作用：** 单写者 SQLite append-only 事件账本，RFC 8785 JSON Canonicalization + SHA-256 digest chaining；一次性许可绑定 effect/intent 序号。结果未知的非可重放操作标记 `TOOL_OUTCOME_UNKNOWN` 并对账，而不是无声重放。
2. **九个强制接口：** approval、principal resolution、billing、workspace checkpoint、artifacts、OS sandbox、completion verification、repair policy、platform capabilities；必备提供者缺失时 fail closed。
3. **快慢决策：** 报告中以 Jev 为例的 Decision Model 在已验证候选下快速选项，Conversation LLM 负责开放规划、填参、异常处理；低信心/超时回交 LLM，执行仍经过内核与设备安全门。
4. **治理型 RSI：** 会话内调整与离线 Playbook 演化分离。后者只改 prompt、workflow、事实规则、决策阈值等配置数据；不改模型权重、代码、硬件安全门、grader 与预算。Train/Eval/Sealed 双生仿真 → 人审 → 200 任务试运行 → 统计回滚。
5. **混合系统 benchmark：** 比较对象是不同“模型 + harness”组合，不能将端到端差异全归功于 AGH。

## 论文摘要中的结果与口径

| 指标 | 报告自报结果 | 限制 |
|---|---|---|
| Home-leak 单步 DM vs LLM | 0.53 s vs 7.6 s（约 14.3 倍） | 封闭仿真；摘要说明 n=8/10 steps |
| Luggage-room 每次决策成本 | $0.000091 vs $0.00247（约 27.1 倍） | 各 100 调用，非整任务成本 |
| Mars Base 工具规格消融 | 18/18 vs 13–14/18 | 依赖工具前置条件完整性 |
| sealed Playbook 仿真 | 行李房 9/9、机械臂单/多故障各 10/10、入库 9/9，分别多次重复 | 非真机试验或零样本泛化 |
| AppWorld dev | 91.67% | 端到端 model+harness |
| MHS-Benchmark | 41/100；其中 desktop GUI 9/40 | GUI 为弱项；不能与物理成功率混算 |
| AGH-Benchmark | 34.62% | 内部自动通过率 |

另有 WorkBuddy Code 78.17、BigCodeBench-Hard V1 40.69%、aider-python 83.33%、TraceTrial 71.8、τ-bench text-v1 87.33%、DeepPlanning 87.07；**均仅据官方摘要，尚未直接检查 PDF 的 13 张表和 9 张图。**

## 证据边界

- 双生仿真的时长下降 44%–75% 仅属指定任务封闭重复，未证跨场景/真实部署。
- 报告承认冷启动、工具 schema 敏感性、Post-compaction 规则保持、GUI grounding、跨平台沙箱与 sim-to-real 未解。
- Jev 决策模型与当前 README 推荐的 Agnes Sys1 侧模型不能未经核对视作同一配置。

## 映射

- [AGH 系统报告节点](../../wiki/entities/paper-agnes-harness-rsi-technical-report.md)
- [Agnes Harness 主实体](../../wiki/entities/agnes-harness.md)
- [Sys1 论文节点](../../wiki/entities/paper-agnes-sys1-decision-model-technical-report.md)
