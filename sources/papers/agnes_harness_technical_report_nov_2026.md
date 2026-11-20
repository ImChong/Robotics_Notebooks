# Agnes Harness Technical Report（2026 年 11 月）

> 来源归档，对应 [独立论文详情](../../wiki/entities/paper-agnes-harness-rsi-technical-report.md)。

- **标题：** Agnes Harness Technical Report: A Ledger-Authoritative, Dual-System & Self-Improving Runtime for Production Agents
- **发布方：** Agnes AI Systems Team；2026-11（官方摘要称 PDF 共 21 页）
- **论文 PDF：** https://github.com/AgnesAI-Labs/Agnes-Harness-Technical-Report-Nov-2026/blob/main/PAPER.pdf
- **官方可读摘要：** https://github.com/AgnesAI-Labs/Agnes-Harness-Technical-Report-Nov-2026/blob/main/REPORT.md
- **实现仓库：** https://github.com/AgnesAI-Labs/agnes-harness
- **核对范围：** 已使用原始 PDF 页面文本抽取读取完整 21 页（§1–10），核对图 1–9 的文字标注及表 1–13 的数字与说明，尤其核查 §3 Ledger/Permits、§7 DM/LLM 路由与表 6–7、§8 两层 RSI/Train-Eval-Sealed/表 8–9、§9 九套 benchmark/表 10–13、§10 六项局限；同时通读官方 REPORT.md，对照 AGH README、架构及启动入口。图表通过 PDF 文本层与表内文字核对，未在本地执行模型/硬件复现实验。
- **通读：** 已完成
- **PDF Git blob：** 71fea5aa56a8874a4f5dc5ff25ec49104e9fb9f4（官方仓库的公开文件身份）
- **入库：** 2026-11-20

## 关键机制

1. **账本与副作用：** 单写者 SQLite append-only 事件账本，RFC 8785 JSON Canonicalization + SHA-256 digest chaining；一次性许可绑定 effect/intent 序号。结果未知的非可重放操作标记 `TOOL_OUTCOME_UNKNOWN` 并对账，而不是无声重放。
2. **九个强制接口：** approval、principal resolution、billing、workspace checkpoint、artifacts、OS sandbox、completion verification、repair policy、platform capabilities；必备提供者缺失时 fail closed。
3. **快慢决策：** 报告中以 Jev 为例的 Decision Model 在已验证候选下快速选项，Conversation LLM 负责开放规划、填参、异常处理；低信心/超时回交 LLM，执行仍经过内核与设备安全门。
4. **治理型 RSI：** 会话内调整与离线 Playbook 演化分离。后者只改 prompt、workflow、事实规则、决策阈值等配置数据；不改模型权重、代码、硬件安全门、grader 与预算。Train/Eval/Sealed 双生仿真 → 人审 → 200 任务试运行 → 统计回滚。
5. **混合系统 benchmark：** 比较对象是不同“模型 + harness”组合，不能将端到端差异全归功于 AGH。

## 原始 PDF 的实验结果与口径

| 指标 | 报告自报结果 | 限制 |
|---|---|---|
| Home-leak 单步 DM vs LLM | 0.53 s vs 7.6 s（约 14.3 倍） | 封闭仿真；摘要说明 n=8/10 steps |
| Luggage-room 每次决策成本 | $0.000091 vs $0.00247（约 27.1 倍） | 各 100 调用，非整任务成本 |
| Mars Base 工具规格消融 | 18/18 vs 13–14/18 | 依赖工具前置条件完整性 |
| sealed Playbook 仿真 | 行李房 9/9、机械臂单/多故障各 10/10、入库 9/9，分别多次重复 | 非真机试验或零样本泛化 |
| AppWorld dev | 91.67% | 端到端 model+harness |
| MHS-Benchmark | 41/100；其中 desktop GUI 9/40 | GUI 为弱项；不能与物理成功率混算 |
| AGH-Benchmark | 34.62% | 内部自动通过率 |

原始 PDF 表 10（p18）给出全部九套系统评测：WorkBuddy Code 78.17（80 任务、每任务 3 次）；BigCodeBench-Hard V1 40.69%；aider-python 83.33%（低于 Codex 91.67%）；TraceTrial-Bench 71.8（harness-level 指标，不能与模型工具调用通过率混为一谈）；τ-bench text-v1 87.33%；DeepPlanning 87.07。表 13（p21）将 MHS 的 100 个任务拆为 desktop GUI 9/40 和 multi-site physical operations 32/60，合计 41/100。**报告声称 100 real operational tasks，但此处仍仅为发布团队的评测数据，不是独立真机复现或认证。**

## 原始报告进一步说明的实现细节

- **双系统策略（§7.2–7.4，图 6）：** 用户刚说话、上一步失败、主动设备检查、DM 连续执行 3 步等情形直交 LLM。DM 用完整设备工具前置条件建立候选，2,000 ms 超时或低信心/完全重复时交还 LLM。Mode A 直接执行，Mode B 由 DM 选动作、LLM 填自由参数，Mode C 完全交 LLM；三者始终经相同的内核/Hub 安全门。已标 correct 的 LLM 动作可进入 DM Option Library；自由值跨 3 次已验证运行相同后才固定，schema 更新或人工复核推翻时撤销。
- **双层 RSI（§8）：** Tier 1 会话范围的 refine 默认关闭，在 turn 空闲边界提议至多 5 项受约束改动；Tier 2 离线跨任务只改可版本化 Playbook 数据。默认 2 代 × 3 候选，对每个案例多次 twin-world 仿真；Train、Eval、Sealed 切分，候选不许更改 grader，发布需人工签字并有 200 任务在线 probation 回滚。
- **任务标签优先级（图 7，p16）：** 人工急停/接管 → wrong；基础设施断连 → uncertain；任务开始前冻结的期望状态与结束状态对比；后续用户重复或纠正 → wrong。人审和来源准确性筛选可撤销误标。
- **架构细节：** PDF 自报 70 packages、46 边界校验套件、2367 测试文件；这些是 2026-11 文档快照，不能直接推成仓库任意后续 commit 的实际数量。

## 证据边界

- 双生仿真的时长下降 44%–75% 仅属指定任务封闭重复，未证跨场景/真实部署。
- 报告承认冷启动、工具 schema 敏感性、Post-compaction 规则保持、GUI grounding、跨平台沙箱与 sim-to-real 未解。
- Jev 决策模型与当前 README 推荐的 Agnes Sys1 侧模型不能未经核对视作同一配置。

## 映射

- [AGH 系统报告节点](../../wiki/entities/paper-agnes-harness-rsi-technical-report.md)
- [Agnes Harness 主实体](../../wiki/entities/agnes-harness.md)
- [Sys1 论文节点](../../wiki/entities/paper-agnes-sys1-decision-model-technical-report.md)
